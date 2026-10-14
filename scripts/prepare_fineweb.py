"""Download and tokenise a slice of FineWeb-Edu into train/val ``.bin`` files.

FineWeb-Edu is a 1.3T-token web corpus filtered for educational quality. It is
the dataset behind several of the strongest small open models, and GPT-2-sized
models trained on it beat the original GPT-2 on HellaSwag with 10x fewer tokens.

We stream the ``sample-10BT`` subset (no full download), tokenise with GPT-2's
BPE (``tiktoken``) in parallel processes, separate documents with the
end-of-text token, and stop after ``--tokens`` tokens.

Usage:  python scripts/prepare_fineweb.py --tokens 1_000_000_000 --out data/fineweb
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

ENC = None


def _init_worker() -> None:
    """Each worker process builds its own tokenizer (tiktoken objects can't be pickled)."""
    global ENC
    import tiktoken

    ENC = tiktoken.get_encoding("gpt2")


def tokenize(text: str) -> np.ndarray:
    """Document -> [<|endoftext|>, tokens...] as uint16; the EOT token marks document boundaries."""
    ids = [ENC.eot_token] + ENC.encode_ordinary(text)
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1_000_000_000, help="training tokens to write")
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--out", default="data/fineweb")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = (row["text"] for row in stream)

    # The first val_tokens go to validation (documents never seen in training), the rest to train.
    targets = [("val", args.val_tokens), ("train", args.tokens)]
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        docs = pool.imap(tokenize, texts, chunksize=64)
        for split, budget in targets:
            # Pre-allocate the output file and fill it; avoids holding 2 GB of lists in RAM.
            arr = np.memmap(out / f"{split}.bin", dtype=np.uint16, mode="w+", shape=(budget,))
            filled = 0
            for ids in docs:
                take = min(len(ids), budget - filled)
                arr[filled : filled + take] = ids[:take]
                filled += take
                if filled >= budget:
                    break
            arr.flush()
            print(f"{split}: {filled:,} tokens -> {out / f'{split}.bin'}", flush=True)


if __name__ == "__main__":
    main()
