"""Single-process tests: model correctness, LR schedule, data windows, MFU math, CLI parsing."""

import math

import numpy as np
import pytest
import torch

from disttrain.data import TokenWindows, write_tokens
from disttrain.metrics import ThroughputMeter, flops_per_token, peak_flops
from disttrain.model import build_model
from disttrain.train import TrainConfig, lr_at, parse_args


def test_forward_shapes_and_loss_at_init():
    torch.manual_seed(0)
    model = build_model("tiny")
    x = torch.randint(0, 256, (3, 16))
    y = torch.randint(0, 256, (3, 16))  # independent targets (with y = x, weight tying makes copying easy)
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 256)
    # A freshly initialised model is close to uniform over the vocabulary: loss ~ ln(256).
    assert loss.item() == pytest.approx(math.log(256), rel=0.05)


def test_causal_mask_future_tokens_do_not_leak():
    torch.manual_seed(0)
    model = build_model("tiny").eval()
    x = torch.randint(0, 256, (1, 20))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 256  # change only the future
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    # Positions 0..9 can't see positions >= 10, so their logits must be identical.
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-6)
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_weight_tying_and_param_count():
    model = build_model("gpt2-small")
    assert model.lm_head.weight is model.wte.weight
    # GPT-2 small: ~124M including position embeddings (with the padded 50304 vocab).
    assert 123e6 < model.num_params(non_embedding=False) < 125e6


def test_activation_checkpointing_gives_identical_gradients():
    torch.manual_seed(0)
    plain = build_model("tiny")
    ckpt = build_model("tiny", activation_checkpointing=True)
    ckpt.load_state_dict(plain.state_dict())
    x = torch.randint(0, 256, (2, 32))
    for m in (plain, ckpt):
        m.train()
        m(x, x)[1].backward()
    for (n, p1), p2 in zip(plain.named_parameters(), ckpt.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6), n


def test_model_can_overfit_a_tiny_batch():
    """Sanity check of the whole optimisation path: loss must collapse on one repeated batch."""
    torch.manual_seed(0)
    model = build_model("tiny")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (4, 32))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_lr_schedule_warmup_peak_and_floor():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(60, cfg) == pytest.approx(0.55)  # halfway through the cosine
    assert lr_at(110, cfg) == pytest.approx(0.1)
    values = [lr_at(s, cfg) for s in range(10, 110)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # monotone decay after warm-up


def test_token_windows_shift_and_rank_streams(tmp_path):
    path = write_tokens(np.arange(1000) % 500, tmp_path / "t.bin")
    a = TokenWindows(path, block_size=8, rank=0, seed=1)
    b = TokenWindows(path, block_size=8, rank=1, seed=1)
    xa, ya = a.batch(4, torch.device("cpu"))
    xb, _ = b.batch(4, torch.device("cpu"))
    assert torch.equal((xa + 1) % 500, ya)  # target = input shifted by one
    assert not torch.equal(xa, xb)  # different ranks see different data
    val = list(TokenWindows(path, 8).sequential_batches(4, torch.device("cpu"), max_batches=3))
    assert len(val) == 3 and torch.equal(val[0][0][0], torch.arange(8))


def test_write_tokens_rejects_large_ids(tmp_path):
    with pytest.raises(ValueError):
        write_tokens([70000], tmp_path / "x.bin")


def test_mfu_math():
    n, L, T, d = 124_000_000, 12, 1024, 768
    assert flops_per_token(n, L, T, d) == 6 * n + 12 * L * T * d
    assert peak_flops("NVIDIA A100-SXM4-40GB") == 312e12
    assert peak_flops("Tesla T4") == 65e12
    meter = ThroughputMeter(flops_per_token=1e9, peak_flops_total=1e12, warmup_steps=1)
    meter.step(100)  # warm-up step: ignored
    assert meter.tokens_per_sec == 0.0


def test_cli_overrides_yaml(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("preset: tiny\nmicro_batch_size: 4\ncompile: false\n")
    cfg = parse_args(["--config", str(cfg_file), "--micro_batch_size", "8", "--compile", "true", "--lr", "1e-3"])
    assert (cfg.preset, cfg.micro_batch_size, cfg.compile, cfg.lr) == ("tiny", 8, True, 1e-3)
