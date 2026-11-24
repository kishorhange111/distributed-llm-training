"""Packed streaming dataloader with sequence packing and assistant-only loss masking.

Packs multiple variable-length instruction/chat sequences into fixed-length ``seq_len``
chunks so GPU utilization is not wasted on padding tokens, while preserving:

* **Assistant-only loss masking**: system and user tokens are masked with ``ignore_index``
  so cross-entropy loss is computed strictly on assistant completion tokens.
* **Document-boundary position resets**: ``position_ids`` restart at 0 for each packed
  document inside a chunk.
* **Block-diagonal causal mask**: ``doc_ids`` allow building a causal mask that prevents
  tokens from attending across packed document boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import torch

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    """One turn in a multi-turn conversation, already tokenized into integer IDs."""

    role: Role
    token_ids: list[int]


@dataclass(frozen=True)
class PackedSample:
    """One fixed-length packed sequence ready to be collated into a batch."""

    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    doc_ids: list[int]


@dataclass(frozen=True)
class PackedBatch:
    """Collated batch of packed sequences."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    doc_ids: torch.Tensor

    def to(self, device: torch.device) -> PackedBatch:
        return PackedBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            position_ids=self.position_ids.to(device),
            doc_ids=self.doc_ids.to(device),
        )


def encode_chat_turns(
    turns: list[ChatTurn],
    *,
    eos_token_id: int,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    """Build shifted ``(input_ids, labels)`` with assistant-only loss masking.

    Every non-assistant token is masked with ``ignore_index`` in ``labels``.
    For assistant turns, each token at position ``t`` predicts token ``t + 1``,
    and the final assistant token predicts ``eos_token_id``.
    """
    tokens: list[int] = []
    loss_mask: list[bool] = []
    for turn in turns:
        if not turn.token_ids:
            continue
        tokens.extend(turn.token_ids)
        is_assistant = turn.role == "assistant"
        loss_mask.extend([is_assistant] * len(turn.token_ids))
        if is_assistant:
            tokens.append(eos_token_id)
            loss_mask.append(True)

    if len(tokens) < 2:
        return [], []

    input_ids = tokens[:-1]
    target_tokens = tokens[1:]
    target_mask = loss_mask[1:]
    labels = [tok if keep else ignore_index for tok, keep in zip(target_tokens, target_mask)]
    return input_ids, labels


def build_packed_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a boolean ``(B, 1, T, T)`` block-diagonal causal attention mask.

    Position ``i`` may attend to position ``j`` iff ``j <= i``, both belong to the
    same packed document (``doc_ids[b, i] == doc_ids[b, j]``), and neither is padding
    (``doc_ids >= 0``).
    """
    if doc_ids.ndim == 1:
        doc_ids = doc_ids.unsqueeze(0)
    batch_size, seq_len = doc_ids.shape
    device = doc_ids.device
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    valid = (doc_ids >= 0).unsqueeze(2) & (doc_ids >= 0).unsqueeze(1)
    mask = causal.unsqueeze(0) & same_doc & valid
    return mask.view(batch_size, 1, seq_len, seq_len)


class PackedStreamingDataset:
    """Streams and packs variable-length ``(input_ids, labels)`` examples into fixed ``seq_len`` chunks."""

    def __init__(
        self,
        examples: Iterable[tuple[list[int], list[int]]],
        *,
        seq_len: int,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank={rank} out of bounds for world_size={world_size}")
        self.examples = examples
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.total_tokens = 0
        self.non_pad_tokens = 0

    @property
    def packing_efficiency(self) -> float:
        """Fraction of emitted tokens that are real (non-padding) tokens."""
        return self.non_pad_tokens / self.total_tokens if self.total_tokens else 0.0

    def __iter__(self) -> Iterator[PackedSample]:
        buf_in: list[int] = []
        buf_lbl: list[int] = []
        buf_pos: list[int] = []
        buf_doc: list[int] = []
        doc_counter = 0

        for idx, (inp, lbl) in enumerate(self.examples):
            if idx % self.world_size != self.rank:
                continue
            if len(inp) != len(lbl):
                raise ValueError(f"input_ids length ({len(inp)}) != labels length ({len(lbl)})")
            if not inp:
                continue

            # Truncate individual examples longer than seq_len.
            inp = inp[: self.seq_len]
            lbl = lbl[: self.seq_len]
            pos = list(range(len(inp)))
            docs = [doc_counter] * len(inp)
            doc_counter += 1

            # Flush current buffer if adding this example would overflow seq_len.
            if buf_in and len(buf_in) + len(inp) > self.seq_len:
                yield self._pad_and_emit(buf_in, buf_lbl, buf_pos, buf_doc)
                buf_in, buf_lbl, buf_pos, buf_doc = [], [], [], []

            buf_in.extend(inp)
            buf_lbl.extend(lbl)
            buf_pos.extend(pos)
            buf_doc.extend(docs)

            if len(buf_in) == self.seq_len:
                yield self._pad_and_emit(buf_in, buf_lbl, buf_pos, buf_doc)
                buf_in, buf_lbl, buf_pos, buf_doc = [], [], [], []

        if buf_in and not self.drop_last:
            yield self._pad_and_emit(buf_in, buf_lbl, buf_pos, buf_doc)

    def _pad_and_emit(
        self,
        inp: list[int],
        lbl: list[int],
        pos: list[int],
        doc: list[int],
    ) -> PackedSample:
        real_len = len(inp)
        pad_len = self.seq_len - real_len
        self.non_pad_tokens += real_len
        self.total_tokens += self.seq_len
        if pad_len > 0:
            inp = inp + [self.pad_token_id] * pad_len
            lbl = lbl + [self.ignore_index] * pad_len
            pos = pos + [0] * pad_len
            doc = doc + [-1] * pad_len
        return PackedSample(input_ids=inp, labels=lbl, position_ids=pos, doc_ids=doc)


class PackedDataLoader:
    """Batches :class:`PackedSample` items from a :class:`PackedStreamingDataset` into :class:`PackedBatch`."""

    def __init__(self, dataset: PackedStreamingDataset, batch_size: int, drop_last: bool = False) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[PackedBatch]:
        chunk: list[PackedSample] = []
        for sample in self.dataset:
            chunk.append(sample)
            if len(chunk) == self.batch_size:
                yield self._collate(chunk)
                chunk = []
        if chunk and not self.drop_last:
            yield self._collate(chunk)

    @staticmethod
    def _collate(samples: list[PackedSample]) -> PackedBatch:
        return PackedBatch(
            input_ids=torch.tensor([s.input_ids for s in samples], dtype=torch.long),
            labels=torch.tensor([s.labels for s in samples], dtype=torch.long),
            position_ids=torch.tensor([s.position_ids for s in samples], dtype=torch.long),
            doc_ids=torch.tensor([s.doc_ids for s in samples], dtype=torch.long),
        )
