from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config import EmeraConfig


def encode_utf8_parity_tokens(text: str) -> list[int]:
    out: list[int] = []
    parity = 0
    for b in text.encode("utf-8", errors="ignore"):
        out.append(int(b) + 256 * parity)
        parity ^= 1
    return out


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_gpt2_tokenizer(model_name: str):
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is not None:
        return tok
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gpt2 token space requires `transformers` with tokenizer support."
        ) from exc
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # We use tokenizer-only mode (no model forward), so long texts are valid.
    if hasattr(tok, "model_max_length"):
        tok.model_max_length = int(10**12)
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def encode_gpt2_tokens(text: str, model_name: str) -> list[int]:
    tok = _get_gpt2_tokenizer(model_name)
    ids = tok.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def decode_gpt2_tokens(token_ids: Iterable[int], model_name: str) -> str:
    tok = _get_gpt2_tokenizer(model_name)
    ids = [int(x) for x in token_ids]
    if not ids:
        return ""
    return tok.decode(ids, clean_up_tokenization_spaces=False)


def _build_world_from_corpus(path: Path, target_len: int, cfg: EmeraConfig) -> np.ndarray:
    data = path.read_text(encoding="utf-8", errors="ignore")
    if cfg.token_space == "gpt2":
        tokens = encode_gpt2_tokens(data, cfg.gpt2_model_name)
    else:
        tokens = encode_utf8_parity_tokens(data)
    if not tokens:
        raise ValueError(f"Corpus produced no tokens: {path}")
    arr = np.asarray(tokens, dtype=np.int32)
    if cfg.token_space == "gpt2" and cfg.base_tokens > 0:
        arr = np.clip(arr, 0, cfg.base_tokens - 1).astype(np.int32)
    if arr.size >= target_len:
        return arr[:target_len]
    reps = int(np.ceil(float(target_len) / float(arr.size)))
    return np.tile(arr, reps)[:target_len]


def _build_zipf_world(cfg: EmeraConfig, rng: np.random.Generator) -> np.ndarray:
    # Structured Zipf world: repeated scaffold motifs with variable Zipf content.
    vocab_lim = int(min(cfg.world_vocab_size, cfg.base_tokens))
    top = np.arange(min(24, vocab_lim), dtype=np.int32)
    tail = np.arange(min(24, vocab_lim), vocab_lim, dtype=np.int32)
    if tail.size == 0:
        tail = top.copy()

    # Create a small motif bank where scaffolds recur and content changes.
    motifs: list[np.ndarray] = []
    for _ in range(64):
        f = rng.choice(top, size=4, replace=True)
        c1 = tail[(int(rng.zipf(a=cfg.zipf_alpha)) - 1) % tail.size]
        c2 = tail[(int(rng.zipf(a=cfg.zipf_alpha)) - 1) % tail.size]
        motif = np.array([f[0], f[1], c1, f[2], c2, f[3]], dtype=np.int32)
        motifs.append(motif)

    world = np.zeros((cfg.world_len,), dtype=np.int32)
    pos = 0
    while pos < cfg.world_len:
        motif = motifs[int(rng.integers(0, len(motifs)))]
        burst = int(rng.integers(2, 8))
        for _ in range(burst):
            local = motif.copy()
            # Small content variation preserves structure while keeping predictability.
            if rng.random() < 0.65:
                local[2] = tail[(int(rng.zipf(a=cfg.zipf_alpha)) - 1) % tail.size]
            if rng.random() < 0.65:
                local[4] = tail[(int(rng.zipf(a=cfg.zipf_alpha)) - 1) % tail.size]
            n = min(local.size, cfg.world_len - pos)
            world[pos : pos + n] = local[:n]
            pos += n
            if pos >= cfg.world_len:
                break
    return world


@dataclass
class World:
    tokens: np.ndarray
    cursor: int
    total_distance: int
    counts: np.ndarray

    @classmethod
    def create(cls, cfg: EmeraConfig, rng: np.random.Generator) -> "World":
        if cfg.corpus_file:
            world_tokens = _build_world_from_corpus(Path(cfg.corpus_file), cfg.world_len, cfg)
        else:
            world_tokens = _build_zipf_world(cfg, rng)

        counts = np.bincount(world_tokens, minlength=cfg.base_tokens).astype(np.float64)
        return cls(tokens=world_tokens, cursor=0, total_distance=0, counts=counts)

    @property
    def length(self) -> int:
        return int(self.tokens.size)

    def _indices(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.zeros((0,), dtype=np.int64)
        idx = (self.cursor + np.arange(length, dtype=np.int64)) % self.tokens.size
        return idx

    def peek(self, length: int) -> np.ndarray:
        return self.tokens[self._indices(length)]

    def advance(self, length: int) -> None:
        length = max(0, int(length))
        self.cursor = int((self.cursor + length) % self.tokens.size)
        self.total_distance += length

    def match_prefix_len(self, proposed: Iterable[int]) -> int:
        seq = np.asarray(list(proposed), dtype=np.int32)
        if seq.size == 0:
            return 0
        gt = self.peek(seq.size)
        mismatches = np.where(gt != seq)[0]
        if mismatches.size == 0:
            return int(seq.size)
        return int(mismatches[0])

    def rarity_of_sequence(self, seq: Iterable[int]) -> float:
        arr = np.asarray(list(seq), dtype=np.int32)
        if arr.size == 0:
            return 0.0
        n = float(self.tokens.size)
        denom = np.log(n + 1.0)
        if denom <= 0.0:
            return 0.0
        freqs = self.counts[np.clip(arr, 0, self.counts.size - 1)]
        idf = np.log((n + self.counts.size) / (freqs + 1.0)) / denom
        return float(np.clip(np.mean(idf), 0.0, 2.0))
