"""MLflow experiment tracking callback and automated post-training evaluation hook.

Both components are rank-aware (logging only on rank 0) and work offline in CPU CI
without requiring a live MLflow server, while forwarding directly to ``mlflow`` when
installed and configured.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TrainingCallback:
    """Base interface for training lifecycle hooks."""

    def on_train_start(self, params: dict[str, Any]) -> None:
        pass

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_eval_end(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, summary: dict[str, Any], model: Any | None = None) -> None:
        pass


class CallbackList(TrainingCallback):
    """Dispatches lifecycle events to an ordered list of callbacks."""

    def __init__(self, callbacks: list[TrainingCallback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def on_train_start(self, params: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(params)

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(step, metrics)

    def on_eval_end(self, step: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(step, metrics)

    def on_train_end(self, summary: dict[str, Any], model: Any | None = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(summary, model=model)


class MLflowCallback(TrainingCallback):
    """Logs parameters, step metrics, and artifacts to MLflow (or a local JSON fallback).

    Non-main ranks (``is_main=False``) are no-ops so distributed runs never duplicate
    tracking records.
    """

    def __init__(
        self,
        experiment_name: str,
        *,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        fallback_dir: str | Path | None = None,
        is_main: bool = True,
        mlflow_module: Any | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name or "disttrain-run"
        self.tracking_uri = tracking_uri
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None
        self.is_main = is_main
        self._mlflow = mlflow_module
        self._active = False
        self.params: dict[str, Any] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self.status: str = "PENDING"

    def _resolve_mlflow(self) -> Any | None:
        if self._mlflow is not None:
            return self._mlflow
        if self.tracking_uri is None:
            return None
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            return None
        return mlflow

    def on_train_start(self, params: dict[str, Any]) -> None:
        if not self.is_main:
            return
        self.params = dict(params)
        self._active = True
        self.status = "RUNNING"
        client = self._resolve_mlflow()
        if client is not None:
            if self.tracking_uri:
                client.set_tracking_uri(self.tracking_uri)
            client.set_experiment(self.experiment_name)
            client.start_run(run_name=self.run_name)
            client.log_params({k: str(v) for k, v in params.items()})
        self._flush_local()

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        if not self.is_main or not self._active:
            return
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        self.metrics_history.append({"step": step, **clean})
        client = self._resolve_mlflow()
        if client is not None and clean:
            client.log_metrics(clean, step=step)

    def on_eval_end(self, step: int, metrics: dict[str, float]) -> None:
        self.on_step_end(step, {f"eval/{k}": v for k, v in metrics.items()})

    def log_artifact(self, path: str | Path) -> None:
        if not self.is_main or not self._active:
            return
        path_str = str(path)
        self.artifacts.append(path_str)
        client = self._resolve_mlflow()
        if client is not None:
            client.log_artifact(path_str)
        self._flush_local()

    def on_train_end(self, summary: dict[str, Any], model: Any | None = None) -> None:
        if not self.is_main or not self._active:
            return
        numeric = {f"final/{k}": float(v) for k, v in summary.items() if isinstance(v, (int, float)) and math.isfinite(v)}
        if numeric:
            self.metrics_history.append({"step": int(summary.get("final_step", 0)), **numeric})
        client = self._resolve_mlflow()
        if client is not None:
            if numeric:
                client.log_metrics(numeric)
            client.end_run()
        self.status = "FINISHED"
        self._active = False
        self._flush_local()

    def _flush_local(self) -> None:
        if self.fallback_dir is None:
            return
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "status": self.status,
            "params": self.params,
            "metrics": self.metrics_history,
            "artifacts": self.artifacts,
        }
        (self.fallback_dir / "mlflow_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class EvalThresholds:
    """Automated quality & performance regression gates for post-training evaluation."""

    max_val_loss: float | None = None
    max_perplexity: float | None = None
    min_tokens_per_sec: float | None = None


@dataclass(frozen=True)
class PostTrainEvalResult:
    """Outcome of running the automated post-train evaluation hook."""

    val_loss: float
    perplexity: float
    passed: bool
    failures: list[str] = field(default_factory=list)
    extra_metrics: dict[str, float] = field(default_factory=dict)


class PostTrainEvalHook(TrainingCallback):
    """Runs evaluation automatically at the end of training and enforces regression gates."""

    def __init__(
        self,
        eval_fn: Callable[[Any], dict[str, float]],
        *,
        thresholds: EvalThresholds | None = None,
        report_path: str | Path | None = None,
        mlflow_callback: MLflowCallback | None = None,
        is_main: bool = True,
    ) -> None:
        self.eval_fn = eval_fn
        self.thresholds = thresholds or EvalThresholds()
        self.report_path = Path(report_path) if report_path else None
        self.mlflow_callback = mlflow_callback
        self.is_main = is_main
        self.last_result: PostTrainEvalResult | None = None

    def run(self, model: Any, summary: dict[str, Any] | None = None) -> PostTrainEvalResult:
        metrics = self.eval_fn(model)
        if "val_loss" not in metrics:
            raise KeyError("eval_fn must return a dict containing 'val_loss'")
        val_loss = float(metrics["val_loss"])
        perplexity = math.exp(min(val_loss, 20.0))
        extra = {k: float(v) for k, v in metrics.items() if k != "val_loss"}

        failures: list[str] = []
        if self.thresholds.max_val_loss is not None and val_loss > self.thresholds.max_val_loss:
            failures.append(f"val_loss {val_loss:.4f} > max_val_loss {self.thresholds.max_val_loss:.4f}")
        if self.thresholds.max_perplexity is not None and perplexity > self.thresholds.max_perplexity:
            failures.append(f"perplexity {perplexity:.2f} > max_perplexity {self.thresholds.max_perplexity:.2f}")
        if self.thresholds.min_tokens_per_sec is not None and summary is not None:
            tps = float(summary.get("tokens_per_sec", 0.0))
            if tps < self.thresholds.min_tokens_per_sec:
                failures.append(f"tokens_per_sec {tps:.1f} < min_tokens_per_sec {self.thresholds.min_tokens_per_sec:.1f}")

        result = PostTrainEvalResult(
            val_loss=round(val_loss, 6),
            perplexity=round(perplexity, 4),
            passed=not failures,
            failures=failures,
            extra_metrics=extra,
        )
        self.last_result = result

        if self.is_main and self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")
            if self.mlflow_callback is not None:
                self.mlflow_callback.on_eval_end(
                    int((summary or {}).get("final_step", 0)),
                    {"val_loss": result.val_loss, "perplexity": result.perplexity, **extra},
                )
                self.mlflow_callback.log_artifact(self.report_path)
        return result

    def on_train_end(self, summary: dict[str, Any], model: Any | None = None) -> None:
        self.run(model, summary)
