"""modallabs.models.diffusion -- minimal DDPM trainer (toy 2D for smoke).

A tiny 2D-feature denoising diffusion model so the smoke test can run on
CPU in seconds. The same Trainer also accepts cfg.image_size for image
data when run on a GPU; for the smoke we stick with 2D.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modallabs.base import (
    Trainer, TrainerEpochResult, TrainerSetup, TrainerStepResult,
)
from modallabs.registry import register

from modallabs.models._torch_common import mean_metrics, resolve_device


@register("diffusion")
class DiffusionTrainer(Trainer):
    """Minimal DDPM-style noise predictor for smoke testing the framework."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.feat_dim = int(self.config.get("feat_dim", 2))
        self.hidden_dim = int(self.config.get("hidden_dim", 64))
        self.timesteps = int(self.config.get("timesteps", 50))
        self.lr = float(self.config.get("lr", 1e-3))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.epochs = int(self.config.get("epochs", 1))
        self.n_samples = int(self.config.get("n", 512))
        self.device = "cpu"
        self.model = None
        self.opt = None
        self.X_train = None
        self.X_val = None
        self._betas = None
        self._alphas = None
        self._alpha_bars = None
        self._train_buf: List[Dict[str, float]] = []
        self._eval_buf: List[Dict[str, float]] = []
        self._best: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DiffusionTrainer":
        return cls(config)

    def setup(self, setup: TrainerSetup) -> None:
        import torch
        import torch.nn as nn

        self.device = resolve_device(setup.device)
        feat = self.feat_dim

        class _Net(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(feat + 1, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, feat),
                )

            def forward(self, x, t_norm):
                return self.net(torch.cat([x, t_norm.unsqueeze(-1)], dim=-1))

        self.model = _Net(self.hidden_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Linear beta schedule
        self._betas = torch.linspace(1e-4, 2e-2, self.timesteps, device=self.device)
        self._alphas = 1.0 - self._betas
        self._alpha_bars = torch.cumprod(self._alphas, dim=0)

        # Synthetic 2-mode dataset
        g = torch.Generator().manual_seed(int(setup.seed))
        modes = torch.tensor([[2.0, 0.0], [-2.0, 0.0]], dtype=torch.float32)
        idx = torch.randint(0, 2, (self.n_samples,), generator=g)
        X = modes[idx] + 0.3 * torch.randn(self.n_samples, self.feat_dim, generator=g)
        cut = max(1, int(self.n_samples * 0.9))
        self.X_train = X[:cut].to(self.device)
        self.X_val = X[cut:].to(self.device) if cut < self.n_samples else self.X_train

    def _sample_noise(self, X):
        import torch
        B = X.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=X.device)
        a_bar = self._alpha_bars[t].unsqueeze(-1)
        eps = torch.randn_like(X)
        x_t = a_bar.sqrt() * X + (1.0 - a_bar).sqrt() * eps
        return x_t, eps, t.float() / self.timesteps

    def _iter(self, X) -> Iterable[Any]:
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i: i + self.batch_size]

    def train_iter(self) -> Iterable[Any]:
        self._train_buf.clear()
        self.model.train()
        return self._iter(self.X_train)

    def eval_iter(self) -> Iterable[Any]:
        self._eval_buf.clear()
        self.model.eval()
        return self._iter(self.X_val)

    def train_step(self, batch: Any) -> TrainerStepResult:
        import torch.nn.functional as F
        Xb = batch
        x_t, eps, t_norm = self._sample_noise(Xb)
        pred = self.model(x_t, t_norm)
        loss = F.mse_loss(pred, eps)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        m = {"loss": float(loss.item())}
        self._train_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(Xb.shape[0]))

    def eval_step(self, batch: Any) -> TrainerStepResult:
        import torch
        import torch.nn.functional as F
        Xb = batch
        with torch.no_grad():
            x_t, eps, t_norm = self._sample_noise(Xb)
            pred = self.model(x_t, t_norm)
            loss = F.mse_loss(pred, eps)
        m = {"loss": float(loss.item())}
        self._eval_buf.append(m)
        return TrainerStepResult(metrics=m, n_examples=int(Xb.shape[0]))

    def epoch_summary(self, epoch: int) -> TrainerEpochResult:
        train_m = mean_metrics(self._train_buf)
        val_m = mean_metrics(self._eval_buf)
        monitor = -float(val_m.get("loss", float("inf")))
        is_best = self._best is None or monitor > self._best
        if is_best:
            self._best = monitor
        return TrainerEpochResult(
            train_metrics=train_m, val_metrics=val_m,
            is_best=is_best, monitor_value=monitor,
        )

    def save_checkpoint(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
