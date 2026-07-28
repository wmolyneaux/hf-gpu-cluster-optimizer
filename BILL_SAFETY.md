# Bill safety: timeout lanes and hard termination

Added 2026-07-25. Motivation: the harness was pinned to a single `(H100, 1800s)` remote function, so
jobs longer than 30 minutes could not run at all — but simply lengthening the timeout raises the
worst-case bill proportionally, because **the timeout *is* the cost bound**
(`worst_case = timeout x rate`, which is exactly what `estimate_total_cost_usd()` charges).

So lanes were added *together with* four independent termination layers. Each layer alone bounds the
spend; they are ordered outermost (unbypassable) to innermost (cheapest to trip).

## The four layers

| # | Layer | Where | Bypassable? |
|---|---|---|---|
| **L1** | `_ABSOLUTE_MAX_USD = 100.0` — a ceiling on the ceiling | `_max_total_usd()` | **No.** Not by env, config, or flag. Only a reviewed code change. |
| **L2** | Modal `timeout=` — server-side container kill | `@app.function` | No. Modal enforces it even if the container is wedged and ignoring us. |
| **L3** | Deadline watchdog, fires `_DEADLINE_MARGIN_SEC` (120s) before L2 | `_remote_body` | In-container, so a total kernel hang falls through to L2. |
| **L4** | Dead-man switch — terminates after `_NO_PROGRESS_KILL_SEC` (600s) with no forward progress | `_remote_body` | Same as L3. |

**L4 is the one that saves real money.** L1-L3 bound the *maximum*; L4 stops you *paying* it. Without
it a wedged job silently bills the entire lane — a hung 4-hour H100 run costs about **$15.80 instead
of the ~$2 it should have**. L1-L3 would all report "within budget" while that happened.

### L4 does not require trainer cooperation

The obvious design is a `progress_cb` passed into the trainer. It was rejected: `train_one` takes no
such parameter, and adding one would make the guard depend on every current and future trainer
remembering to call it. Instead L4 polls the **newest mtime anywhere under the run directory**. Every
trainer writes checkpoints and logs, so mtime is the one progress signal that is true for all of them
and cannot be forgotten.

Startup gets a separate, larger budget (`_STARTUP_GRACE_SEC = 900`) because cold start, image pull and
HuggingFace download all precede the first write. Too tight kills a legitimate cold start; too loose
lets a job that dies at import burn the grace period.

`os._exit()` is used deliberately. A wedged trainer may be stuck inside a C extension where an
exception can never be delivered, and the entire point of this thread is that it works **anyway**. The
container dies, Modal stops billing, and the run resumes from its last checkpoint.

## Lanes

Every lane is additional worst-case exposure, so the table is deliberately short and `main()` always
routes a run to the **smallest lane that fits** its requested `max_runtime_sec`.

| Lane | Timeout | Gate accounting (stale $5.50/h table) | Actual invoice ($0.001097/s) |
|---|---|---|---|
| `short` (default) | 1800s / 30 min | $2.75 | **$1.97** |
| `medium` | 5400s / 90 min | $8.25 | **$5.92** |
| `long` | 14400s / 4 h | $22.00 | **$15.80** |

A request exceeding the largest lane is **refused**, not silently rounded up. Fix by splitting the
job, checkpointing and resuming, or adding a lane deliberately with review.

The two dollar columns are both correct and must never be mixed: the **gate** computes with the
harness's stale in-file price table (so it decides whether the launcher blocks), while the **invoice**
uses the rate verified against modal.com/pricing. `modal_app.py:55-57` says itself that the table
needs verifying.

## Behaviour changes

- Mixed **timeouts** in one sweep now **route** instead of raising. Mixed **GPUs** still raise, since
  every lane is pinned to `_REMOTE_GPU`.
- The cost ceiling is **re-asserted immediately before spend**, not only during the dry-run. This
  catches a config mutated between preflight and launch, at the last point where nothing is billed.
- Lane routing and the armed guard limits are printed before launch.
- `MODALLABS_MAX_USD` can only ever **lower** the gate. A value above `_ABSOLUTE_MAX_USD` is clamped
  and warned about — a typo in an env var must not be able to authorise an unbounded bill.
- `_REMOTE_TIMEOUT_SEC` is kept as a back-compat alias; existing configs are unaffected and the
  default lane reproduces the previous behaviour exactly.

## Verification (all run 2026-07-25, nothing spent)

```
syntax                    parses clean
MODALLABS_MAX_USD=10   -> ceiling $10.00      (env can lower)
MODALLABS_MAX_USD=99999-> ceiling $100.00     (CLAMPED + warned)
MODALLABS_MAX_USD=-5   -> ceiling $25.00      (rejects nonsense)
MODALLABS_MAX_USD=junk -> ceiling $25.00
lane routing  600s->short  1800s->short  1801s->medium  5401s->long  20000s->REFUSED
dry-run       $1.04 within $25 ceiling, exit 0
imports       modallabs OK, 25 trainers registered
preflight     PASS=12 FAIL=0 WARN=0 -> PROCEED (exit 0)
```

L4 was verified against a simulated hang rather than by inspection:

```
healthy (writes every 0.8s)      -> no_kill            correct
hangs after making progress      -> L4_KILL_idle       correct
hangs at startup (never writes)  -> L4_KILL_startup    correct
```

## Also fixed: a false-positive HALT in preflight CODE-5

`check_modal_app_antipatterns` string-matched the raw source, so a **comment** documenting that
`Function.with_options()` does *not* exist tripped the check and produced `verdict: HALT` on a correct
file. It now strips comments and string literals via `tokenize` before matching (tokenize rather than
a regex, so a `#` inside a string literal is not mistaken for a comment), falling back to raw text if
the file will not parse so a genuine antipattern is never silently skipped.

This was pre-existing — `git show HEAD:modal_app.py` contains the same comment. It mattered because a
gate that cries wolf gets ignored, which is worse than no gate.
