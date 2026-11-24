"""CI smoke tests for Kubernetes PyTorchJob / TorchElastic manifests and Dockerfile."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_defines_non_root_trainer_and_entrypoint() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER trainer" in dockerfile
    assert "disttrain.train" in dockerfile


def test_pytorchjob_fsdp_manifest_structure() -> None:
    doc = yaml.safe_load((ROOT / "deploy" / "k8s" / "pytorchjob-fsdp.yaml").read_text(encoding="utf-8"))
    assert doc["apiVersion"] == "kubeflow.org/v1"
    assert doc["kind"] == "PyTorchJob"
    specs = doc["spec"]["pytorchReplicaSpecs"]
    assert "Master" in specs and "Worker" in specs
    for role in ("Master", "Worker"):
        container = specs[role]["template"]["spec"]["containers"][0]
        assert container["command"][0] == "torchrun"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
        mount_paths = {m["mountPath"] for m in container["volumeMounts"]}
        assert {"/checkpoints", "/dev/shm"} <= mount_paths


def test_torchelastic_manifest_structure() -> None:
    doc = yaml.safe_load((ROOT / "deploy" / "k8s" / "torchelastic-job.yaml").read_text(encoding="utf-8"))
    assert doc["apiVersion"] == "kubeflow.org/v1"
    assert doc["kind"] == "PyTorchJob"
    elastic = doc["spec"]["elasticPolicy"]
    assert elastic["rdzvBackend"] == "c10d"
    assert elastic["minReplicas"] >= 1
    assert elastic["maxReplicas"] >= elastic["minReplicas"]
    assert elastic["maxRestarts"] >= 1
    worker = doc["spec"]["pytorchReplicaSpecs"]["Worker"]["template"]["spec"]
    assert worker["containers"][0]["command"][0] == "torchrun"
    assert any(t["key"] == "cloud.google.com/gke-preemptible" for t in worker["tolerations"])
