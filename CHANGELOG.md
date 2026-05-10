# Changelog

All notable changes to `hf_cluster_optimizer` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-10

Initial public release.

### Added
- **Concurrent training orchestrator** (`concurrent_train.py`). One config,
  many models. Each run lands in its own subprocess for CUDA / OOM /
  segfault isolation. Local execution defaults to one process per GPU,
  falls back to `cpu_count()/2`.
- **Modal Labs cloud orchestrator** (`modal_app.py`). Same orchestrator
  config runs on Modal. One `@app.function(...)` per run with a
  per-run GPU type and a Modal-enforced hard timeout. No warm-pool
  keep-alive: containers tear down on completion.
- **Cost controls** (default-on). Pre-flight `--dry-run` preview prints
  worst-case cost per run (`max_runtime_sec * GPU rate`), aggregated
  total, and the configured `HFCO_MAX_USD` ceiling. The launcher
  refuses to spawn any GPU container if the worst-case total breaches
  the ceiling. CLI exit code 2 on breach for `set -e` / CI gates.
- **HuggingFace-forward**. 12 task heads via one `hf_transformer`
  wrapper: `hf_sequence_classification`, `hf_causal_lm`, `hf_seq2seq`,
  `hf_image_classification`, `hf_token_classification`, `hf_qa`,
  `hf_embedding`, `hf_masked_lm`, `hf_multiple_choice`,
  `hf_object_detection`, `hf_audio_classification`, `hf_whisper`.
  Checkpoints save via `save_pretrained` so they round-trip with
  `push_to_hub` directly.
- **Persistent HF cache on Modal**. Second `modal.Volume`
  (`hfco-hf-cache`) mounted at `/hf_cache` with the standard four
  HuggingFace env vars (`HF_HOME`, `TRANSFORMERS_CACHE`,
  `HUGGINGFACE_HUB_CACHE`, `HF_DATASETS_CACHE`) all pointing at it.
  First cold start populates the cache; every subsequent cold start
  reuses it.
- **Crash isolation**. A model that segfaults / OOMs / raises only
  kills its own subprocess. Other models keep training; the
  orchestrator records `phase: failed` and a traceback in
  `status.json` and continues.
- **Resumability**. `--resume` skips runs whose `.hfco_done`
  sentinel exists AND whose checkpoint is verifiably on disk
  (resolved across known suffixes: `.pt` / `.pth` / `.joblib` /
  `.txt` / `.json` / `.cbm` / `.safetensors` / bare-name
  save_pretrained dir). Stale sentinels (no checkpoint) trigger a
  fresh re-run with a stderr warning. On Modal, resume uses a tiny
  CPU-only function so already-done runs don't allocate any GPU.
- **Determinism**. `set_global_seed(N)` covers `random`, `numpy`,
  `torch` (CPU + CUDA), and `transformers`, plus
  `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `cudnn.deterministic=True`. Smoke test asserts identical metrics
  within `1e-6` across two seeded runs on CPU. GPU determinism is
  best-effort (cuDNN / CUDA-version dependent).
- **`Ctrl-C` cancels in-flight futures**. Both local and Modal
  orchestrators catch `KeyboardInterrupt`, cancel pending futures,
  and signal in-flight workers. Containers tear down and GPUs
  release on exit.
- **Defensive `torch.cuda.empty_cache()` after teardown**. Any in-process
  Trainer that holds tensors in long-lived attributes is reclaimable on
  the next run.
- **Duplicate-name detection**. Two runs sharing a `name:` field would
  silently clobber each other's `runs/<run_id>/<name>/`. The orchestrator
  now refuses to start when duplicates or blank names are detected at
  config-load time.
- **Path-typed cfg values are yaml-safe**. `_yaml_safe` recursively
  coerces `pathlib.Path`, `datetime`, `set`, and unknown classes into
  primitives so `config_resolved.yaml` never crashes the run.
- **Unknown GPU type warns once**. Setting `cfg.modal.gpu: B200` (or any
  tier missing from the price table) prints a one-time stderr warning
  naming the substitute rate so the dry-run preview is never silently
  optimistic.
- **Bad `--log-level` warns once**. Typos like `--log-level NOPE` print
  a stderr warning and fall back to INFO instead of silently failing.
- **CI on Ubuntu / macOS / Windows x Python 3.10 / 3.11 / 3.12**.
  Smoke + orchestrator + Modal dry-run all run on every push.

### Supported architectures (out of the box)

- BERT-style encoders (BERT, RoBERTa, DeBERTa)
- GPT / Llama / Mistral causal LMs
- T5 / BART / Pegasus seq2seq
- ViT / Swin / ConvNeXt image classification
- Whisper / Wav2Vec2 audio
- Token classification, QA, embeddings, masked LM, multiple choice,
  object detection
- Generic `torch.nn.Module` (any class, dotted-path import)
- Generic sklearn estimator
- LightGBM, XGBoost, CatBoost
- DDPM-style diffusion (toy bundled; swap your own UNet via
  `torch_module`)
- DQN-style offline Q-learning with double-Q + dueling head

[0.1.0]: https://github.com/<owner>/hf-cluster-optimizer/releases/tag/v0.1.0
