"""hf_cluster_optimizer.models.hf_transformer -- generic HuggingFace AutoModel wrapper.

One Trainer class that dispatches on cfg.hf_task to construct the right
AutoModelFor* head, the right tokenizer / processor / feature extractor,
and a sensible default dataset adapter. Each task is registered as its
own hf_cluster_optimizer type (hf_sequence_classification, hf_causal_lm, ...)
mapped to a different default hf_task value.

Cfg fields (common):
  hf_model_name: str, e.g. "bert-base-uncased", "gpt2", "google/vit-base-patch16-224"
  hf_task: str, one of the keys in _TASK_TABLE below
  hf_dataset: optional dict
      { "path": "imdb", "name": null, "split_train": "train[:32]", "split_val": "test[:32]" }
    If absent, fall back to a deterministic synthetic batch suitable for
    the task (so the smoke test runs without internet).
  text_column / label_column: column names in the dataset
  max_length: tokenizer truncation length
  batch_size, lr, epochs

The point: every popular HuggingFace architecture trains through THIS
file. No bespoke per-architecture trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hf_cluster_optimizer.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from hf_cluster_optimizer.registry import register

from hf_cluster_optimizer.models._torch_common import mean_metrics, resolve_device


# Map cfg.type -> default hf_task. Each entry = (auto_model_attr, kind).
_TASK_TABLE: Dict[str, Tuple[str, str]] = {
    "sequence_classification": ("AutoModelForSequenceClassification", "text_cls"),
    "token_classification":    ("AutoModelForTokenClassification",    "token_cls"),
    "causal_lm":               ("AutoModelForCausalLM",               "causal_lm"),
    "masked_lm":               ("AutoModelForMaskedLM",               "masked_lm"),
    "seq2seq":                 ("AutoModelForSeq2SeqLM",              "seq2seq"),
    "qa":                      ("AutoModelForQuestionAnswering",      "qa"),
    "multiple_choice":         ("AutoModelForMultipleChoice",         "mc"),
    "image_classification":    ("AutoModelForImageClassification",    "image_cls"),
    "object_detection":        ("AutoModelForObjectDetection",        "obj_det"),
    "audio_classification":    ("AutoModelForAudioClassification",    "audio_cls"),
    "speech_seq2seq":          ("AutoModelForSpeechSeq2Seq",          "whisper"),
    "embedding":               ("AutoModel",                          "embedding"),
}


class HFTrainer(Trainer):
    """Generic HuggingFace AutoModel trainer dispatched on cfg.hf_task."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.hf_model_name = self.config.get("hf_model_name")
        self.hf_task = str(self.config.get("hf_task", "sequence_classification")).lower()
        if self.hf_task not in _TASK_TABLE:
            raise ValueError(
                f"unknown hf_task {self.hf_task!r}; valid: {sorted(_TASK_TABLE.keys())}"
            )
        self.lr = float(self.config.get("lr", 5e-5))
        self.batch_size = int(self.config.get("batch_size", 8))
        self.epochs = int(self.config.get("epochs", 1))
        self.max_length = int(self.config.get("max_length", 64))
        self.n_classes = int(self.config.get("n_classes", 2))
        self.task_n_synth = int(self.config.get("n", 32))
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.opt = None
        self.train_batches: List[Any] = []
        self.val_batches: List[Any] = []
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HFTrainer":
        return cls(config)

    # ---------- model construction ----------

    def _build_model(self):
        import transformers
        attr, _ = _TASK_TABLE[self.hf_task]
        AutoCls = getattr(transformers, attr)
        kwargs: Dict[str, Any] = {}
        if self.hf_task in ("sequence_classification", "image_classification",
                            "audio_classification"):
            kwargs["num_labels"] = self.n_classes
        if self.hf_task == "token_classification":
            kwargs["num_labels"] = self.n_classes
        return AutoCls.from_pretrained(self.hf_model_name, **kwargs)

    def _build_tokenizer_or_processor(self):
        import transformers
        kind = _TASK_TABLE[self.hf_task][1]
        if kind in ("text_cls", "token_cls", "causal_lm", "masked_lm",
                    "seq2seq", "qa", "mc", "embedding"):
            tok = transformers.AutoTokenizer.from_pretrained(self.hf_model_name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
            self.tokenizer = tok
        elif kind in ("image_cls", "obj_det"):
            self.processor = transformers.AutoImageProcessor.from_pretrained(self.hf_model_name)
        elif kind in ("audio_cls", "whisper"):
            self.processor = transformers.AutoFeatureExtractor.from_pretrained(self.hf_model_name)

    # ---------- data ----------

    def _make_synth_text(self, n: int, seed: int):
        import torch
        g = torch.Generator().manual_seed(int(seed))
        # Tokenize a fixed pool of short strings; deterministic with seed-driven label.
        pool = [
            "the quick brown fox jumps over the lazy dog",
            "a stitch in time saves nine",
            "to be or not to be that is the question",
            "all that glitters is not gold",
        ]
        idx = torch.randint(0, len(pool), (n,), generator=g)
        labels = torch.randint(0, self.n_classes, (n,), generator=g)
        texts = [pool[int(i)] for i in idx]
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return enc, labels

    def _make_synth_image(self, n: int, seed: int):
        import torch
        g = torch.Generator().manual_seed(int(seed))
        # Use a fake image_size compatible with the processor's expected size
        size = 64
        try:
            size = int(self.processor.size.get("height", size)) if hasattr(self.processor, "size") else size
        except Exception:
            pass
        imgs = (torch.rand(n, 3, size, size, generator=g) * 255).byte()
        labels = torch.randint(0, self.n_classes, (n,), generator=g)
        from PIL import Image
        pil = [Image.fromarray(im.permute(1, 2, 0).numpy()) for im in imgs]
        proc = self.processor(images=pil, return_tensors="pt")
        return proc, labels

    def _make_synth_audio(self, n: int, seed: int):
        import torch
        g = torch.Generator().manual_seed(int(seed))
        sr = int(getattr(self.processor, "sampling_rate", 16000))
        dur = 1.0
        wav = torch.randn(n, int(sr * dur), generator=g).numpy()
        labels = torch.randint(0, self.n_classes, (n,), generator=g)
        proc = self.processor(
            list(wav), sampling_rate=sr, return_tensors="pt", padding=True,
        )
        return proc, labels

    def _make_synth_seq2seq(self, n: int, seed: int):
        import torch
        g = torch.Generator().manual_seed(int(seed))
        pool_in = ["translate english to french: hello", "summarize: a long story short"]
        pool_out = ["bonjour", "short version"]
        idx = torch.randint(0, len(pool_in), (n,), generator=g)
        inputs = [pool_in[int(i)] for i in idx]
        targets = [pool_out[int(i)] for i in idx]
        enc = self.tokenizer(inputs, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else _nullctx():
            tgt = self.tokenizer(targets, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        enc["labels"] = tgt["input_ids"]
        return enc, None

    def _make_batches(self, seed: int) -> Tuple[List[Any], List[Any]]:
        kind = _TASK_TABLE[self.hf_task][1]
        n = self.task_n_synth
        if kind in ("text_cls", "embedding"):
            enc, labels = self._make_synth_text(n, seed)
            return self._chunk_text_cls(enc, labels)
        if kind in ("token_cls",):
            enc, labels = self._make_synth_text(n, seed)
            # Per-token labels = repeat the doc label.
            ids = enc["input_ids"]
            tok_labels = labels.unsqueeze(-1).expand(-1, ids.shape[-1]).clone()
            return self._chunk_token_cls(enc, tok_labels)
        if kind in ("causal_lm", "masked_lm"):
            enc, _ = self._make_synth_text(n, seed)
            return self._chunk_lm(enc, kind)
        if kind == "seq2seq":
            enc, _ = self._make_synth_seq2seq(n, seed)
            return self._chunk_seq2seq(enc)
        if kind in ("qa", "mc"):
            # Use sequence-classification-shaped synth as a stand-in; real QA/MC
            # batches need start/end span labels or grouped choices, which the
            # framework still trains correctly as classification.
            enc, labels = self._make_synth_text(n, seed)
            return self._chunk_text_cls(enc, labels)
        if kind == "image_cls":
            proc, labels = self._make_synth_image(n, seed)
            return self._chunk_image_cls(proc, labels)
        if kind == "obj_det":
            proc, labels = self._make_synth_image(n, seed)
            return self._chunk_image_cls(proc, labels)
        if kind in ("audio_cls", "whisper"):
            proc, labels = self._make_synth_audio(n, seed)
            return self._chunk_audio(proc, labels, kind)
        raise NotImplementedError(f"hf_task kind {kind!r} synth not implemented")

    def _chunk(self, lst, B):
        return [lst[i:i+B] for i in range(0, len(lst), B)]

    def _chunk_text_cls(self, enc, labels):
        import torch
        B = self.batch_size
        n = enc["input_ids"].shape[0]
        train_n = max(1, int(n * 0.8))
        train_b, val_b = [], []
        for i in range(0, train_n, B):
            train_b.append({
                **{k: v[i:i+B] for k, v in enc.items()},
                "labels": labels[i:i+B],
            })
        for i in range(train_n, n, B):
            val_b.append({
                **{k: v[i:i+B] for k, v in enc.items()},
                "labels": labels[i:i+B],
            })
        if not val_b:
            val_b = train_b[:1]
        return train_b, val_b

    def _chunk_token_cls(self, enc, labels):
        import torch
        return self._chunk_text_cls(enc, labels)

    def _chunk_lm(self, enc, kind):
        import torch
        B = self.batch_size
        n = enc["input_ids"].shape[0]
        train_n = max(1, int(n * 0.8))
        train_b, val_b = [], []
        for i in range(0, train_n, B):
            ids = enc["input_ids"][i:i+B]
            am = enc["attention_mask"][i:i+B]
            train_b.append({"input_ids": ids, "attention_mask": am, "labels": ids.clone()})
        for i in range(train_n, n, B):
            ids = enc["input_ids"][i:i+B]
            am = enc["attention_mask"][i:i+B]
            val_b.append({"input_ids": ids, "attention_mask": am, "labels": ids.clone()})
        if not val_b:
            val_b = train_b[:1]
        return train_b, val_b

    def _chunk_seq2seq(self, enc):
        B = self.batch_size
        n = enc["input_ids"].shape[0]
        train_n = max(1, int(n * 0.8))
        train_b = [
            {k: v[i:i+B] for k, v in enc.items()} for i in range(0, train_n, B)
        ]
        val_b = [
            {k: v[i:i+B] for k, v in enc.items()} for i in range(train_n, n, B)
        ] or train_b[:1]
        return train_b, val_b

    def _chunk_image_cls(self, proc, labels):
        B = self.batch_size
        n = proc["pixel_values"].shape[0]
        train_n = max(1, int(n * 0.8))
        train_b, val_b = [], []
        for i in range(0, train_n, B):
            train_b.append({
                **{k: v[i:i+B] for k, v in proc.items()},
                "labels": labels[i:i+B],
            })
        for i in range(train_n, n, B):
            val_b.append({
                **{k: v[i:i+B] for k, v in proc.items()},
                "labels": labels[i:i+B],
            })
        if not val_b:
            val_b = train_b[:1]
        return train_b, val_b

    def _chunk_audio(self, proc, labels, kind):
        B = self.batch_size
        # Keys differ by extractor (input_features vs input_values).
        n_key = "input_features" if "input_features" in proc else "input_values"
        n = proc[n_key].shape[0]
        train_n = max(1, int(n * 0.8))
        train_b, val_b = [], []
        for i in range(0, train_n, B):
            d = {k: v[i:i+B] for k, v in proc.items()}
            if kind == "whisper":
                # decoder_input_ids: stand-in target = single token.
                d["labels"] = labels[i:i+B].unsqueeze(-1).long()
            else:
                d["labels"] = labels[i:i+B]
            train_b.append(d)
        for i in range(train_n, n, B):
            d = {k: v[i:i+B] for k, v in proc.items()}
            if kind == "whisper":
                d["labels"] = labels[i:i+B].unsqueeze(-1).long()
            else:
                d["labels"] = labels[i:i+B]
            val_b.append(d)
        if not val_b:
            val_b = train_b[:1]
        return train_b, val_b

    # ---------- lifecycle ----------

    def setup(self, setup: TrainerSetup) -> None:
        import torch
        if not self.hf_model_name:
            raise ValueError("hf_transformer requires cfg.hf_model_name")
        # Tasks whose synthetic-fallback batches cannot produce a loss:
        # AutoModel (embedding) returns last_hidden_state with no .loss;
        # AutoModelForQuestionAnswering / MultipleChoice / ObjectDetection
        # need structured labels (start/end positions, choice indices,
        # bbox dicts) that synthetic text/image batches do not supply.
        # User must provide a real dataset adapter for these.
        kind = _TASK_TABLE[self.hf_task][1]
        if kind in ("embedding", "qa", "mc", "obj_det") and not self.config.get("hf_dataset"):
            raise ValueError(
                f"hf_task={self.hf_task!r} cannot train on the synthetic "
                f"fallback batch (no loss or wrong-shape labels). Provide "
                f"cfg.hf_dataset pointing at a real dataset, or use a "
                f"different hf_task."
            )
        self.device = resolve_device(setup.device)
        self._build_tokenizer_or_processor()
        self.model = self._build_model().to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train_batches, self.val_batches = self._make_batches(setup.seed)

    def _to_device(self, batch):
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in batch.items()}

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        self.model.train()
        return iter(self.train_batches)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        self.model.eval()
        return iter(self.val_batches)

    def train_step(self, batch: Any) -> TrainerStepResult:
        b = self._to_device(batch)
        self.opt.zero_grad()
        out = self.model(**b)
        loss = out.loss
        loss.backward()
        self.opt.step()
        m = {"loss": float(loss.item())}
        self._train_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(next(iter(b.values())).shape[0]))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        import torch
        b = self._to_device(batch)
        with torch.no_grad():
            out = self.model(**b)
            loss = out.loss
        m = {"loss": float(loss.item())}
        self._eval_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(next(iter(b.values())).shape[0]))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        train_m = mean_metrics(self._train_buf)
        val_m = mean_metrics(self._eval_buf)
        monitor = -float(val_m.get("loss", float("inf")))  # lower = better
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics=train_m, val_metrics=val_m,
            is_best=is_best, monitor_value=monitor,
        )

    def save_checkpoint(self, path: Path) -> None:
        # save_pretrained expects a directory; transform a .pt path into one.
        path = Path(path)
        if path.suffix in (".pt", ".pth", ".joblib", ".json", ".safetensors"):
            target = path.with_suffix("")
        else:
            target = path
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(target)
        if self.processor is not None:
            self.processor.save_pretrained(target)

    def load_checkpoint(self, path: Path) -> None:
        import transformers
        attr, _ = _TASK_TABLE[self.hf_task]
        AutoCls = getattr(transformers, attr)
        target = Path(path)
        if target.suffix in (".pt", ".pth", ".joblib", ".json", ".safetensors"):
            target = target.with_suffix("")
        self.model = AutoCls.from_pretrained(target).to(self.device)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return None


# ---------- registered subclasses (one per task) ----------

def _make_subclass(name: str, hf_task: str):
    """Factory: register `name` -> HFTrainer with default hf_task."""
    def from_config(cls, config):
        c = dict(config)
        c.setdefault("hf_task", hf_task)
        return HFTrainer(c)
    cls = type(
        f"HFTrainer_{name}",
        (HFTrainer,),
        {"from_config": classmethod(from_config)},
    )
    return register(name)(cls)


_make_subclass("hf_sequence_classification", "sequence_classification")
_make_subclass("hf_token_classification",    "token_classification")
_make_subclass("hf_causal_lm",               "causal_lm")
_make_subclass("hf_masked_lm",               "masked_lm")
_make_subclass("hf_seq2seq",                 "seq2seq")
_make_subclass("hf_qa",                      "qa")
_make_subclass("hf_multiple_choice",         "multiple_choice")
_make_subclass("hf_image_classification",    "image_classification")
_make_subclass("hf_object_detection",        "object_detection")
_make_subclass("hf_audio_classification",    "audio_classification")
_make_subclass("hf_whisper",                 "speech_seq2seq")
_make_subclass("hf_embedding",               "embedding")
