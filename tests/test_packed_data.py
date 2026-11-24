"""Tests for packed streaming dataloader, sequence packing, and assistant-only loss masking."""

from __future__ import annotations

import pytest
import torch

from disttrain.model import build_model
from disttrain.packed_data import (
    ChatTurn,
    PackedDataLoader,
    PackedStreamingDataset,
    build_packed_causal_mask,
    encode_chat_turns,
)


def test_encode_chat_turns_masks_non_assistant_tokens() -> None:
    turns = [
        ChatTurn("system", [10, 11]),
        ChatTurn("user", [20, 21, 22]),
        ChatTurn("assistant", [30, 31]),
    ]
    input_ids, labels = encode_chat_turns(turns, eos_token_id=99, ignore_index=-100)
    # Full sequence: [10, 11, 20, 21, 22, 30, 31, 99]
    assert input_ids == [10, 11, 20, 21, 22, 30, 31]
    # Targets shifted by 1: [11, 20, 21, 22, 30, 31, 99] -> only assistant tokens (30, 31, 99) unmasked
    assert labels == [-100, -100, -100, -100, 30, 31, 99]


def test_encode_chat_turns_multi_turn_conversation() -> None:
    turns = [
        ChatTurn("user", [1, 2]),
        ChatTurn("assistant", [3]),
        ChatTurn("user", [4, 5]),
        ChatTurn("assistant", [6, 7]),
    ]
    input_ids, labels = encode_chat_turns(turns, eos_token_id=0, ignore_index=-1)
    assert input_ids == [1, 2, 3, 0, 4, 5, 6, 7]
    assert labels == [-1, 3, 0, -1, -1, 6, 7, 0]


def test_packed_streaming_packs_multiple_docs_and_resets_positions() -> None:
    examples = [
        ([1, 2, 3], [-100, 2, 3]),
        ([4, 5, 6, 7], [-100, -100, 6, 7]),
        ([8, 9], [8, 9]),
    ]
    ds = PackedStreamingDataset(examples, seq_len=8, pad_token_id=0, ignore_index=-100)
    samples = list(ds)
    assert len(samples) == 2
    # First chunk packs doc 0 (len 3) + doc 1 (len 4) + 1 pad
    assert samples[0].input_ids == [1, 2, 3, 4, 5, 6, 7, 0]
    assert samples[0].labels == [-100, 2, 3, -100, -100, 6, 7, -100]
    assert samples[0].position_ids == [0, 1, 2, 0, 1, 2, 3, 0]
    assert samples[0].doc_ids == [0, 0, 0, 1, 1, 1, 1, -1]
    assert ds.packing_efficiency == pytest.approx(9 / 16)


def test_packed_causal_mask_blocks_cross_doc_and_future_attention() -> None:
    doc_ids = torch.tensor([[0, 0, 1, 1, -1]])
    mask = build_packed_causal_mask(doc_ids)
    assert mask.shape == (1, 1, 5, 5)
    m = mask[0, 0]
    # Intra-doc causal allowed
    assert m[0, 0].item() is True
    assert m[1, 0].item() is True
    assert m[0, 1].item() is False  # future token blocked
    # Cross-doc blocked
    assert m[2, 0].item() is False
    assert m[2, 1].item() is False
    assert m[2, 2].item() is True
    assert m[3, 2].item() is True
    # Padding blocked everywhere
    assert not m[4].any().item()
    assert not m[:, 4].any().item()


def test_packed_streaming_shards_across_ranks() -> None:
    examples = [([i, i + 1], [i, i + 1]) for i in range(6)]
    r0 = list(PackedStreamingDataset(examples, seq_len=6, rank=0, world_size=2))
    r1 = list(PackedStreamingDataset(examples, seq_len=6, rank=1, world_size=2))
    assert r0[0].input_ids == [0, 1, 2, 3, 4, 5]
    assert r1[0].input_ids == [1, 2, 3, 4, 5, 6]


def test_packed_dataloader_integrates_with_gpt_forward() -> None:
    turns = [ChatTurn("user", [1, 2, 3, 4]), ChatTurn("assistant", [5, 6, 7])]
    examples = [encode_chat_turns(turns, eos_token_id=9, ignore_index=-1) for _ in range(4)]
    ds = PackedStreamingDataset(examples, seq_len=16, pad_token_id=0, ignore_index=-1)
    loader = PackedDataLoader(ds, batch_size=2)
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch.input_ids.shape == (2, 16)
    model = build_model("tiny")
    logits, loss = model(batch.input_ids, batch.labels)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
