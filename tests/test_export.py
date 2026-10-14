"""The Hugging Face export must produce exactly the same logits as our model."""

import pytest
import torch

pytest.importorskip("transformers")

from disttrain.export import to_hf  # noqa: E402
from disttrain.model import build_model  # noqa: E402


def test_hf_export_matches_logits():
    torch.manual_seed(0)
    ours = build_model("tiny").eval()
    hf = to_hf(ours).eval()
    x = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        a, _ = ours(x)
        b = hf(x).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
