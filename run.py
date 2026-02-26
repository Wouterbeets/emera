#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

from config import EmeraConfig
from engine import EmeraEngine


def build_config_from_args(args: argparse.Namespace) -> EmeraConfig:
    cfg = EmeraConfig()
    updates = {}
    for key in [
        "seed",
        "token_space",
        "gpt2_model_name",
        "base_tokens",
        "corpus_file",
        "world_len",
        "world_vocab_size",
        "zipf_alpha",
        "d_latent",
        "gap_dim",
        "gap_len",
        "gap_read_backend",
        "gap_read_batch_size",
        "num_ifs",
        "chaos_substeps_per_round",
        "k_rounds",
        "proposal_lmax",
        "obligatory_proposals",
        "proposal_min_bet",
        "proposal_bet_unit_scale",
        "proposal_bet_max_energy_frac",
        "proposal_bet_floor_frac",
        "proposal_bet_conf_gain",
        "proposal_frontier_contrast",
        "proposal_frontier_fallback",
        "frontier_rescue_energy",
        "frontier_rescue_noise",
        "frontier_rescue_max_per_step",
        "contrastive_enabled",
        "contrastive_correct_reward",
        "contrastive_wrong_penalty",
        "contrastive_wrong_exp",
        "initial_super_tokens",
        "initial_super_energy",
        "ambient_dissipation",
        "metabolic_tax_rate",
        "base_toll",
        "attempt_cost_base",
        "discovery_cost",
        "min_viable_energy",
        "survivor_grace_steps",
        "survivor_relief_active_frac",
        "survivor_relief_reservoir_frac",
        "conserve_total_energy",
        "strict_energy_budget",
        "energy_inflow_per_step",
        "energy_reservoir_init",
        "energy_reservoir_cap",
        "spawn_cost",
        "self_copy_enabled",
        "self_copy_interval",
        "self_copy_cost",
        "self_copy_min_energy",
        "self_copy_min_match_frac",
        "self_copy_min_score",
        "self_copy_max_per_step",
        "mint_interval",
        "mint_delta",
        "ema_alpha",
        "dynamic_laws",
        "law_update_interval",
        "adaptation_rate",
        "adaptation_signal_decay",
        "season_period",
        "season_strength",
        "season_wave_decay",
        "season_revival_spores",
        "season_revival_energy",
        "log_every",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = val
    if updates.get("token_space") == "gpt2":
        if "base_tokens" not in updates:
            updates["base_tokens"] = 50257
        if "world_vocab_size" not in updates:
            updates["world_vocab_size"] = int(updates["base_tokens"])
    cfg = replace(cfg, **updates)
    cfg.validate()
    return cfg


def format_step_line(stats: dict, snapshot: dict) -> str:
    return (
        f"step={stats['step']} active={stats['active_super']} proposers={stats['proposers']} "
        f"rescue={stats.get('frontier_rescue_spawned', 0)} "
        f"rescue_win={stats.get('winner_from_frontier_rescue', 0)} "
        f"win={stats['winner_id']} conf={stats['winner_conf']:.3f} len={stats['proposal_len']} "
        f"super_token_len={stats.get('winner_super_token_len', 0)} "
        f"match={stats['match_len']} adv={stats['advance_len']} "
        f"R={stats['realized_return']:+.3f} payout={stats['jackpot']:.3f} "
        f"costs(a/d/b)={stats.get('attempt_total_actual', stats['attempt_total']):.3f}/"
        f"{stats['discovery_cost']:.3f}/{stats['base_toll']:.3f} "
        f"contrast(b/p)={stats.get('contrastive_bonus', 0.0):.3f}/{stats.get('contrastive_penalty', 0.0):.3f} "
        f"avg_bet={stats.get('attempt_avg', 0.0):.4f} "
        f"births={stats['births']} copy={stats.get('self_copy_births', 0)} deaths={stats['deaths']} "
        f"laws(at={stats['law_attempt_cost_base']:.3f},jk={stats['law_jackpot_base']:.2f},"
        f"sp={stats['law_spawn_cost']:.3f},pa={stats['law_pareto_alpha']:.2f}) "
        f"reservoir={stats.get('reservoir', 0.0):.2f} drift={stats.get('energy_drift', 0.0):+.4f} "
        f"EPA={snapshot['energy_per_advance']:.4f} glide={snapshot['glide_ratio']:.3f} "
        f"tok_glide={snapshot.get('token_glide_ratio', snapshot['glide_ratio']):.3f} "
        f"gap_cr={snapshot.get('gap_compression_ratio', 1.0):.3f} "
        f"depth={int(snapshot.get('max_symbio_depth', 0))} "
        f"root_frac={snapshot.get('root_only_fraction', 1.0):.3f} "
        f"payout_tot={snapshot.get('payout_total', 0.0):.3f}"
    )


def _safe_text_from_identity(identity_values, max_chars: int, token_space: str) -> str:
    if token_space == "byte_parity":
        raw = bytes(int(b) & 0xFF for b in identity_values)
        text = raw.decode("utf-8", errors="backslashreplace")
    else:
        text = " ".join(str(int(x)) for x in identity_values)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text.encode("unicode_escape").decode("ascii").replace("'", "\\'")


def _lineage_label(parent_a: int, parent_b: int) -> str:
    if parent_a < 0 and parent_b < 0:
        return "root"
    return f"{parent_a},{parent_b}"


def _token_age(engine: EmeraEngine, token) -> int:
    return max(int(engine.step_idx) - int(getattr(token, "birth_step", 0)), 0)


def _format_feature_line(engine: EmeraEngine, label: str, token, metric: str, max_chars: int) -> str:
    age = _token_age(engine, token)
    rs = float(engine.last_resonance_strength.get(token.token_id, 0.0))
    _, conf, _ = engine._raw_decode_for_token(token, rs)
    text = _safe_text_from_identity(
        token.identity_bytes,
        max_chars=max_chars,
        token_space=engine.cfg.token_space,
    )
    lineage = _lineage_label(int(token.parent_a), int(token.parent_b))
    st_len = int(len(token.identity_bytes))
    metric_part = f"{metric} " if metric else ""
    return (
        f"  feat {label} id={int(token.token_id)} {metric_part}age={age} e={float(token.energy):.3f} "
        f"conf={float(conf):.3f} super_token_len={st_len} txt='{text}' lineage={lineage}"
    )


def format_feature_lines(engine: EmeraEngine, stats: dict, k: int, max_chars: int) -> list[str]:
    if k <= 0 or not engine.super_tokens:
        return []
    tokens = list(engine.super_tokens.values())
    out: list[str] = []

    paid = [t for t in tokens if float(getattr(t, "max_paid_bet", 0.0)) > 0.0]
    if paid:
        best_paid = max(
            paid,
            key=lambda t: (float(t.max_paid_bet), float(t.energy), int(t.token_id)),
        )
        out.append(
            _format_feature_line(
                engine,
                "biggest_paid_bet",
                best_paid,
                f"paid_bet={float(best_paid.max_paid_bet):.4f}",
                max_chars,
            )
        )
    else:
        out.append("  feat biggest_paid_bet none")

    silent_id = int(stats.get("longest_silent_turn_id", -1))
    if silent_id >= 0:
        silent_steps = int(stats.get("longest_silent_turn_steps", 0))
        silent_match = int(stats.get("longest_silent_turn_match_len", 0))
        silent_prop_len = int(stats.get("longest_silent_turn_proposal_len", 0))
        silent_conf = float(stats.get("longest_silent_turn_conf", 0.0))
        silent_score = float(stats.get("longest_silent_turn_score", 0.0))
        silent_won = int(stats.get("longest_silent_turn_won", 0))
        silent_len = int(stats.get("longest_silent_turn_token_len", 0))
        silent_parent_a = int(stats.get("longest_silent_turn_parent_a", -1))
        silent_parent_b = int(stats.get("longest_silent_turn_parent_b", -1))
        silent_identity = list(stats.get("longest_silent_turn_identity_bytes", []))
        silent_text = _safe_text_from_identity(
            silent_identity,
            max_chars=max_chars,
            token_space=engine.cfg.token_space,
        )
        silent_lineage = _lineage_label(silent_parent_a, silent_parent_b)
        out.append(
            f"  feat longest_silent_activated_turn id={silent_id} silent_steps={silent_steps} "
            f"match={silent_match}/{silent_prop_len} conf={silent_conf:.3f} score={silent_score:+.3f} "
            f"won={silent_won} super_token_len={silent_len} txt='{silent_text}' lineage={silent_lineage}"
        )
    else:
        out.append("  feat longest_silent_activated_turn none")

    oldest = max(tokens, key=lambda t: (_token_age(engine, t), float(t.energy), -int(t.token_id)))
    out.append(
        _format_feature_line(
            engine,
            "oldest",
            oldest,
            "",
            max_chars,
        )
    )
    return out[:k]


class CorpusLineLocator:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()

    def line_at_token_index(self, token_idx: int) -> str:
        if not self.data:
            return ""
        i = int(token_idx) % len(self.data)
        start = self.data.rfind(b"\n", 0, i) + 1
        end = self.data.find(b"\n", i)
        if end < 0:
            end = len(self.data)
        raw = self.data[start:end].rstrip(b"\r")
        return raw.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emera v2 prototype runner")
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--token-space", type=str, choices=["byte_parity", "gpt2"], default=None)
    p.add_argument("--gpt2-model-name", type=str, default=None)
    p.add_argument("--base-tokens", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--corpus-file", type=str, default=None)
    p.add_argument("--world-len", type=int, default=None)
    p.add_argument("--world-vocab-size", type=int, default=None)
    p.add_argument("--zipf-alpha", type=float, default=None)
    p.add_argument("--d-latent", type=int, default=None)
    p.add_argument("--gap-dim", type=int, default=None)
    p.add_argument("--gap-len", type=int, default=None)
    p.add_argument("--gap-read-backend", type=str, choices=["auto", "numpy", "jax"], default=None)
    p.add_argument("--gap-read-batch-size", type=int, default=None)
    p.add_argument("--num-ifs", type=int, default=None)
    p.add_argument("--chaos-substeps-per-round", type=int, default=None)
    p.add_argument("--k-rounds", type=int, default=None)
    p.add_argument("--proposal-lmax", type=int, default=None)
    p.add_argument("--obligatory-proposals", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--proposal-min-bet", type=float, default=None)
    p.add_argument("--proposal-bet-unit-scale", type=float, default=None)
    p.add_argument("--proposal-bet-max-energy-frac", type=float, default=None)
    p.add_argument("--proposal-bet-floor-frac", type=float, default=None)
    p.add_argument("--proposal-bet-conf-gain", type=float, default=None)
    p.add_argument("--proposal-frontier-contrast", type=int, default=None)
    p.add_argument("--proposal-frontier-fallback", type=int, default=None)
    p.add_argument("--frontier-rescue-energy", type=float, default=None)
    p.add_argument("--frontier-rescue-noise", type=float, default=None)
    p.add_argument("--frontier-rescue-max-per-step", type=int, default=None)
    p.add_argument("--contrastive-enabled", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--contrastive-correct-reward", type=float, default=None)
    p.add_argument("--contrastive-wrong-penalty", type=float, default=None)
    p.add_argument("--contrastive-wrong-exp", type=float, default=None)
    p.add_argument("--initial-super-tokens", type=int, default=None)
    p.add_argument("--initial-super-energy", type=float, default=None)
    p.add_argument("--ambient-dissipation", type=float, default=None)
    p.add_argument("--metabolic-tax-rate", type=float, default=None)
    p.add_argument("--base-toll", type=float, default=None)
    p.add_argument("--attempt-cost-base", type=float, default=None)
    p.add_argument("--discovery-cost", type=float, default=None)
    p.add_argument("--min-viable-energy", type=float, default=None)
    p.add_argument("--survivor-grace-steps", type=int, default=None)
    p.add_argument("--survivor-relief-active-frac", type=float, default=None)
    p.add_argument("--survivor-relief-reservoir-frac", type=float, default=None)
    p.add_argument("--conserve-total-energy", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--strict-energy-budget", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--energy-inflow-per-step", type=float, default=None)
    p.add_argument("--energy-reservoir-init", type=float, default=None)
    p.add_argument("--energy-reservoir-cap", type=float, default=None)
    p.add_argument("--spawn-cost", type=float, default=None)
    p.add_argument("--self-copy-enabled", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--self-copy-interval", type=int, default=None)
    p.add_argument("--self-copy-cost", type=float, default=None)
    p.add_argument("--self-copy-min-energy", type=float, default=None)
    p.add_argument("--self-copy-min-match-frac", type=float, default=None)
    p.add_argument("--self-copy-min-score", type=float, default=None)
    p.add_argument("--self-copy-max-per-step", type=int, default=None)
    p.add_argument("--mint-interval", type=int, default=None)
    p.add_argument("--mint-delta", type=float, default=None)
    p.add_argument("--ema-alpha", type=float, default=None)
    p.add_argument("--dynamic-laws", type=lambda s: s.lower() in {"1", "true", "yes", "on"}, default=None)
    p.add_argument("--law-update-interval", type=int, default=None)
    p.add_argument("--adaptation-rate", type=float, default=None)
    p.add_argument("--adaptation-signal-decay", type=float, default=None)
    p.add_argument("--season-period", type=int, default=None)
    p.add_argument("--season-strength", type=float, default=None)
    p.add_argument("--season-wave-decay", type=float, default=None)
    p.add_argument("--season-revival-spores", type=int, default=None)
    p.add_argument("--season-revival-energy", type=float, default=None)
    p.add_argument("--log-oldest-k", type=int, default=3)
    p.add_argument("--log-oldest-max-chars", type=int, default=18)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.log_every is not None and args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if args.log_oldest_k is not None and args.log_oldest_k < 0:
        raise ValueError("--log-oldest-k must be >= 0")
    if args.log_oldest_max_chars is not None and args.log_oldest_max_chars < 0:
        raise ValueError("--log-oldest-max-chars must be >= 0")

    cfg = build_config_from_args(args)
    engine = EmeraEngine(cfg)
    locator: Optional[CorpusLineLocator] = None
    if cfg.corpus_file:
        corpus_path = Path(cfg.corpus_file)
        if corpus_path.exists():
            try:
                locator = CorpusLineLocator(corpus_path)
            except OSError:
                locator = None

    t0 = time.time()
    print(
        f"Emera v2 start | seed={cfg.seed} | world_len={engine.world.length} | "
        f"initial_super={len(engine.super_tokens)} | k_rounds={cfg.k_rounds} | "
        f"gap_backend={engine.gap.read_backend} | gap_batch={cfg.gap_read_batch_size}"
    )

    log_every = cfg.log_every
    for _ in range(args.steps):
        result = engine.step()
        if (result.stats["step"] % log_every == 0) or (result.stats["step"] == 1):
            snap = engine.metrics.snapshot()
            print(format_step_line(result.stats, snap))
            if locator is not None:
                line = locator.line_at_token_index(int(result.stats.get("cursor", engine.world.cursor)))
                print(f"  text: {line}")
            for line in format_feature_lines(
                engine=engine,
                stats=result.stats,
                k=int(args.log_oldest_k),
                max_chars=int(args.log_oldest_max_chars),
            ):
                print(line)

    snap = engine.metrics.snapshot()
    elapsed = time.time() - t0
    final = {
        "config": asdict(cfg),
        "elapsed_sec": elapsed,
        "active_super": len(engine.super_tokens),
        "cursor": engine.world.cursor,
        "next_token_id": engine.next_token_id,
        "metrics": snap,
    }
    print(
        f"done steps={args.steps} elapsed={elapsed:.2f}s active={final['active_super']} "
        f"EPA={snap['energy_per_advance']:.4f} glide={snap['glide_ratio']:.3f} "
        f"tok_glide={snap.get('token_glide_ratio', snap['glide_ratio']):.3f} "
        f"gap_cr={snap.get('gap_compression_ratio', 1.0):.3f} "
        f"depth={int(snap.get('max_symbio_depth', 0))} "
        f"root_frac={snap.get('root_only_fraction', 1.0):.3f} "
        f"births={snap['births_total']} deaths={snap['deaths_total']}"
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(final, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
