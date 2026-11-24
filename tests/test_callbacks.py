"""Tests for MLflow experiment tracking callback and automated post-train eval hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from disttrain.callbacks import CallbackList, EvalThresholds, MLflowCallback, PostTrainEvalHook


class _FakeMLflow:
    def __init__(self) -> None:
        self.uri: str | None = None
        self.experiment: str | None = None
        self.run_name: str | None = None
        self.params: dict[str, str] = {}
        self.metrics: list[tuple[int | None, dict[str, float]]] = []
        self.artifacts: list[str] = []
        self.ended = False

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str | None = None) -> None:
        self.run_name = run_name

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics.append((step, dict(metrics)))

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def end_run(self) -> None:
        self.ended = True


def test_mlflow_callback_logs_to_client_and_fallback(tmp_path: Path) -> None:
    fake = _FakeMLflow()
    cb = MLflowCallback(
        "gpt2-pretrain",
        run_name="run-001",
        tracking_uri="http://mlflow:5000",
        fallback_dir=tmp_path / "mlflow",
        is_main=True,
        mlflow_module=fake,
    )
    cb.on_train_start({"preset": "tiny", "lr": 1e-3})
    cb.on_step_end(1, {"loss": 4.2, "mfu": 0.45})
    cb.on_eval_end(1, {"val_loss": 4.0})
    cb.on_train_end({"final_step": 1, "final_loss": 4.2})

    assert fake.uri == "http://mlflow:5000"
    assert fake.experiment == "gpt2-pretrain"
    assert fake.run_name == "run-001"
    assert fake.params["preset"] == "tiny"
    assert fake.ended is True

    saved = json.loads((tmp_path / "mlflow" / "mlflow_run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FINISHED"
    assert len(saved["metrics"]) == 3


def test_mlflow_callback_is_noop_on_worker_ranks(tmp_path: Path) -> None:
    fake = _FakeMLflow()
    cb = MLflowCallback("exp", fallback_dir=tmp_path / "worker", is_main=False, mlflow_module=fake)
    cb.on_train_start({"preset": "tiny"})
    cb.on_step_end(1, {"loss": 3.0})
    cb.on_train_end({"final_step": 1})
    assert fake.experiment is None
    assert not (tmp_path / "worker" / "mlflow_run.json").exists()


def test_post_train_eval_hook_passes_and_logs_report(tmp_path: Path) -> None:
    mlflow_cb = MLflowCallback("eval-exp", fallback_dir=tmp_path / "mlflow")
    report = tmp_path / "eval_report.json"

    def fake_eval(_model: Any) -> dict[str, float]:
        return {"val_loss": 2.0, "hellaswag_acc": 0.42}

    eval_hook = PostTrainEvalHook(
        fake_eval,
        thresholds=EvalThresholds(max_val_loss=2.5, max_perplexity=15.0, min_tokens_per_sec=1000.0),
        report_path=report,
        mlflow_callback=mlflow_cb,
    )
    # Run eval hook before ending MLflow run so eval metrics and artifact attach cleanly.
    callbacks = CallbackList([mlflow_cb])
    callbacks.on_train_start({"preset": "tiny"})
    eval_hook.on_train_end({"final_step": 100, "tokens_per_sec": 5000.0}, model=object())
    callbacks.on_train_end({"final_step": 100, "tokens_per_sec": 5000.0})

    assert eval_hook.last_result is not None
    assert eval_hook.last_result.passed is True
    assert eval_hook.last_result.failures == []
    assert report.exists()
    assert str(report) in mlflow_cb.artifacts


def test_post_train_eval_hook_flags_regression_failures() -> None:
    hook = PostTrainEvalHook(
        lambda _: {"val_loss": 4.5},
        thresholds=EvalThresholds(max_val_loss=3.5, max_perplexity=20.0, min_tokens_per_sec=10_000.0),
    )
    result = hook.run(model=None, summary={"tokens_per_sec": 5_000.0})
    assert result.passed is False
    assert len(result.failures) == 3
