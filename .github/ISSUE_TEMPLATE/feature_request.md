---
name: Feature request
about: Propose an addition (a model head, a config knob, a workflow)
title: "[feat] "
labels: enhancement
---

## What

What you'd like to be able to do.

## Why

The use case. If it's a new model type, name the HuggingFace
architecture(s) or library it covers.

## Sketch (optional)

How you'd expect it to look in a config:

```yaml
runs:
  - name: ...
    type: ...
    config:
      ...
```

## Scope check

Does this fit the project scope? (See "out of scope" in
[CONTRIBUTING.md](../../CONTRIBUTING.md) — no per-architecture bespoke
trainers, no multi-node distributed training, no warm-pool keep-alive,
no arbitrary-pickle checkpoints.)
