#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from config import EmeraConfig
from engine import EmeraEngine
from world import encode_gpt2_tokens, encode_utf8_parity_tokens


def _parse_bool(v: str | None, default: bool | None = None) -> bool | None:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _parse_horizons(text: str) -> list[int]:
    vals: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(max(int(part), 1))
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("No valid horizons provided.")
    return uniq


def _encode_text_tokens(text: str, token_space: str, gpt2_model_name: str, base_tokens: int) -> np.ndarray:
    if token_space == "gpt2":
        ids = encode_gpt2_tokens(text, gpt2_model_name)
        arr = np.asarray(ids, dtype=np.int32)
        return np.clip(arr, 0, max(base_tokens - 1, 0)).astype(np.int32)
    ids = encode_utf8_parity_tokens(text)
    return np.asarray(ids, dtype=np.int32)


def _snap_to_newline(raw: bytes, idx: int) -> int:
    idx = int(np.clip(idx, 0, len(raw)))
    if idx <= 0 or idx >= len(raw):
        return idx
    right = raw.find(b"\n", idx, min(len(raw), idx + 4096))
    if right >= 0:
        return int(right + 1)
    left = raw.rfind(b"\n", max(0, idx - 4096), idx)
    if left >= 0:
        return int(left + 1)
    return idx


def _split_corpus_bytes(raw: bytes, train_frac: float, val_frac: float) -> tuple[bytes, bytes, bytes]:
    n = len(raw)
    if n < 64:
        raise ValueError("Corpus is too small to split.")
    i_train = _snap_to_newline(raw, int(n * train_frac))
    i_val = _snap_to_newline(raw, int(n * (train_frac + val_frac)))
    i_train = int(np.clip(i_train, 1, n - 2))
    i_val = int(np.clip(i_val, i_train + 1, n - 1))
    return raw[:i_train], raw[i_train:i_val], raw[i_val:]


def _sample_starts(token_count: int, max_h: int, num_contexts: int, seed: int) -> np.ndarray:
    upper = int(token_count - max_h - 1)
    if upper <= 0:
        return np.zeros((0,), dtype=np.int64)
    all_idx = np.arange(upper, dtype=np.int64)
    if num_contexts <= 0 or num_contexts >= upper:
        return all_idx
    rng = np.random.default_rng(seed)
    pick = np.asarray(rng.choice(upper, size=num_contexts, replace=False), dtype=np.int64)
    pick.sort()
    return pick


def _evaluate_horizon(
    engine: EmeraEngine,
    tokens: np.ndarray,
    horizons: list[int],
    num_contexts: int,
    rollout_len: int,
    top_k: int,
    temperature: float,
    prob_floor: float,
    seed: int,
) -> dict[str, Any]:
    max_h = int(max(horizons))
    starts = _sample_starts(int(tokens.size), max_h=max_h, num_contexts=num_contexts, seed=seed)
    if starts.size == 0:
        raise ValueError("Split too small for requested horizons.")

    per_h: dict[int, dict[str, float]] = {
        int(h): {"nll_sum": 0.0, "acc_sum": 0.0, "count": 0.0}
        for h in horizons
    }
    rollout_n = max(1, min(int(rollout_len), max_h))
    exact_sum = 0.0
    first_err_sum = 0.0

    floor = max(float(prob_floor), 1e-15)
    for start in starts.tolist():
        seq: list[int] = [int(tokens[start])]
        preds: list[int] = []
        dists: list[dict[int, float]] = []
        for _ in range(max_h):
            nxt, dist, _ = engine._infer_next_distribution(
                frontier_tid=seq[-1],
                right_tid=None,
                top_k=int(top_k),
                temperature=float(temperature),
                recent_tokens=seq,
            )
            preds.append(int(nxt))
            dists.append(dist)
            seq.append(int(nxt))

        truth = np.asarray(tokens[start + 1 : start + max_h + 1], dtype=np.int32)
        for h in horizons:
            idx = int(h - 1)
            true_tok = int(truth[idx])
            p = float(dists[idx].get(true_tok, floor))
            p = max(p, floor)
            per_h[h]["nll_sum"] += -math.log(p)
            per_h[h]["acc_sum"] += 1.0 if int(preds[idx]) == true_tok else 0.0
            per_h[h]["count"] += 1.0

        first_err = rollout_n
        exact = 1.0
        for i in range(rollout_n):
            if int(preds[i]) != int(truth[i]):
                first_err = i
                exact = 0.0
                break
        exact_sum += exact
        first_err_sum += float(first_err)

    by_h: dict[str, Any] = {}
    for h in horizons:
        c = max(per_h[h]["count"], 1.0)
        by_h[str(h)] = {
            "acc": per_h[h]["acc_sum"] / c,
            "nll": per_h[h]["nll_sum"] / c,
        }

    long_h = [h for h in horizons if h >= 32]
    if not long_h:
        long_h = horizons[-max(1, len(horizons) // 2) :]

    lr_score = float(np.mean([by_h[str(h)]["acc"] for h in long_h]))
    lr_nll = float(np.mean([by_h[str(h)]["nll"] for h in long_h]))
    contexts = float(starts.size)
    return {
        "contexts": int(starts.size),
        "horizons": by_h,
        "lr_horizons": [int(h) for h in long_h],
        "lr_score": lr_score,
        "lr_nll": lr_nll,
        "rollout_len": int(rollout_n),
        "rollout_exact_match_rate": exact_sum / contexts,
        "rollout_first_error_mean": first_err_sum / contexts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out horizon-conditioned probability benchmark for Emera")
    p.add_argument("--corpus-file", type=str, default="data/bible.txt")
    p.add_argument("--token-space", type=str, choices=["byte_parity", "gpt2"], default="byte_parity")
    p.add_argument("--gpt2-model-name", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--world-vocab-size", type=int, default=512)
    p.add_argument("--world-len", type=int, default=None)
    p.add_argument("--gap-read-backend", type=str, choices=["auto", "numpy", "jax"], default=None)
    p.add_argument("--gap-read-batch-size", type=int, default=None)
    p.add_argument("--horizons", type=str, default="1,2,4,8,16,32,64,128,256")
    p.add_argument("--num-contexts", type=int, default=512)
    p.add_argument("--rollout-len", type=int, default=256)
    p.add_argument("--eval-top-k", type=int, default=0)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--prob-floor", type=float, default=1e-9)
    p.add_argument("--splits-dir", type=str, default="out/horizon_splits")
    p.add_argument("--output-json", type=str, default="out/horizon_eval.json")

    # High-impact knobs for tuning sweeps.
    p.add_argument("--proposal-lmax", type=int, default=None)
    p.add_argument("--proposal-frontier-fallback", type=int, default=None)
    p.add_argument("--proposal-frontier-contrast", type=int, default=None)
    p.add_argument("--frontier-rescue-max-per-step", type=int, default=None)
    p.add_argument("--frontier-rescue-energy", type=float, default=None)
    p.add_argument("--self-copy-enabled", type=str, default=None)
    p.add_argument("--dynamic-laws", type=str, default=None)
    p.add_argument("--seasons-enabled", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    if args.train_steps < 1:
        raise ValueError("--train-steps must be >= 1")
    if args.num_contexts < 1:
        raise ValueError("--num-contexts must be >= 1")
    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be in (0, 1)")
    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val-frac must be in (0, 1)")
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    raw = Path(args.corpus_file).read_bytes()
    train_raw, val_raw, test_raw = _split_corpus_bytes(raw, train_frac=float(args.train_frac), val_frac=float(args.val_frac))
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    test_file = splits_dir / "test.txt"
    train_file.write_bytes(train_raw)
    val_file.write_bytes(val_raw)
    test_file.write_bytes(test_raw)

    train_text = train_raw.decode("utf-8", errors="ignore")
    val_text = val_raw.decode("utf-8", errors="ignore")
    test_text = test_raw.decode("utf-8", errors="ignore")

    base_tokens = 50257 if args.token_space == "gpt2" else 512
    val_tokens = _encode_text_tokens(val_text, args.token_space, args.gpt2_model_name, base_tokens=base_tokens)
    test_tokens = _encode_text_tokens(test_text, args.token_space, args.gpt2_model_name, base_tokens=base_tokens)
    train_tokens = _encode_text_tokens(train_text, args.token_space, args.gpt2_model_name, base_tokens=base_tokens)

    cfg = EmeraConfig()
    world_len = int(args.world_len) if args.world_len is not None else int(max(train_tokens.size, 1))
    updates: dict[str, Any] = {
        "seed": int(args.seed),
        "token_space": str(args.token_space),
        "gpt2_model_name": str(args.gpt2_model_name),
        "corpus_file": str(train_file),
        "world_len": int(world_len),
        "world_vocab_size": int(args.world_vocab_size),
        "log_every": max(int(args.train_steps), 1),
    }
    if args.token_space == "gpt2":
        updates["base_tokens"] = 50257
        if args.world_vocab_size == 512:
            updates["world_vocab_size"] = 50257
    if args.proposal_lmax is not None:
        updates["proposal_lmax"] = int(args.proposal_lmax)
    if args.gap_read_backend is not None:
        updates["gap_read_backend"] = str(args.gap_read_backend)
    if args.gap_read_batch_size is not None:
        updates["gap_read_batch_size"] = int(args.gap_read_batch_size)
    if args.proposal_frontier_fallback is not None:
        updates["proposal_frontier_fallback"] = int(args.proposal_frontier_fallback)
    if args.proposal_frontier_contrast is not None:
        updates["proposal_frontier_contrast"] = int(args.proposal_frontier_contrast)
    if args.frontier_rescue_max_per_step is not None:
        updates["frontier_rescue_max_per_step"] = int(args.frontier_rescue_max_per_step)
    if args.frontier_rescue_energy is not None:
        updates["frontier_rescue_energy"] = float(args.frontier_rescue_energy)
    b_self_copy = _parse_bool(args.self_copy_enabled, default=None)
    if b_self_copy is not None:
        updates["self_copy_enabled"] = bool(b_self_copy)
    b_laws = _parse_bool(args.dynamic_laws, default=None)
    if b_laws is not None:
        updates["dynamic_laws"] = bool(b_laws)
    b_seasons = _parse_bool(args.seasons_enabled, default=None)
    if b_seasons is not None:
        updates["seasons_enabled"] = bool(b_seasons)

    cfg = replace(cfg, **updates)
    cfg.validate()

    t0 = time.time()
    engine = EmeraEngine(cfg)
    for _ in range(int(args.train_steps)):
        engine.step()
    train_elapsed = time.time() - t0
    train_snap = engine.metrics.snapshot()
    train_distance = int(train_snap.get("distance", 0))
    effective_passes = float(train_distance) / max(float(cfg.world_len), 1.0)

    eval_kwargs = dict(
        horizons=horizons,
        num_contexts=int(args.num_contexts),
        rollout_len=int(args.rollout_len),
        top_k=int(args.eval_top_k),
        temperature=float(args.eval_temperature),
        prob_floor=float(args.prob_floor),
    )
    val_eval = _evaluate_horizon(
        engine=engine,
        tokens=val_tokens,
        seed=int(args.seed) + 11,
        **eval_kwargs,
    )
    test_eval = _evaluate_horizon(
        engine=engine,
        tokens=test_tokens,
        seed=int(args.seed) + 29,
        **eval_kwargs,
    )

    out = {
        "config": asdict(cfg),
        "train_steps": int(args.train_steps),
        "train_elapsed_sec": float(train_elapsed),
        "train_distance": int(train_distance),
        "effective_train_passes": float(effective_passes),
        "split_bytes": {
            "train": int(len(train_raw)),
            "val": int(len(val_raw)),
            "test": int(len(test_raw)),
        },
        "split_tokens": {
            "train": int(train_tokens.size),
            "val": int(val_tokens.size),
            "test": int(test_tokens.size),
        },
        "val": val_eval,
        "test": test_eval,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        f"horizon-eval done train_steps={args.train_steps} "
        f"val_lr={val_eval['lr_score']:.4f} val_lr_nll={val_eval['lr_nll']:.4f} "
        f"test_lr={test_eval['lr_score']:.4f} test_lr_nll={test_eval['lr_nll']:.4f} "
        f"wrote={out_path}"
    )


if __name__ == "__main__":
    main()
