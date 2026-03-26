"""Minimal script to load a shard and decode tokens back to text."""

from pathlib import Path
import numpy as np
import sentencepiece as spm

SHARD = Path("data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin")
TOKENIZER = "data/tokenizers/fineweb_1024_bpe.model"

# Load shard: 256-int header followed by uint16 token ids
header = np.fromfile(SHARD, dtype="<i4", count=256)
num_tokens = int(header[2])
tokens = np.fromfile(SHARD, dtype="<u2", count=num_tokens, offset=256 * 4)

sp = spm.SentencePieceProcessor(model_file=TOKENIZER)

print(f"Shard: {SHARD}")
print(f"Vocab size: {sp.vocab_size()}")
print(f"Total tokens in shard: {num_tokens:,}")
print()

cnt = 1024
# Show first cnt tokens: ids and decoded text
ids = tokens[:cnt].tolist()
print(f"First {cnt} token IDs:\n{ids}\n")
print(f"Decoded text:\n{sp.decode(ids)}\n")

# Show per-token breakdown for the first 20 tokens
print("Per-token breakdown (first 20):")
for i, tid in enumerate(ids[:20]):
    piece = sp.id_to_piece(tid)
    print(f"  [{i:2d}] id={tid:4d}  piece={piece!r}")
