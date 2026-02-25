from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import EmeraConfig
from cooperation import CooperationStats
from gap_field import GapField
from genome import (
    Proposal,
    SuperToken,
    create_initial_population,
    make_initial_super_token,
    mint_child_from_parent,
    mint_child_from_parents,
)
from identity import BaseIdentity, create_base_identity
from ledger import (
    attempt_cost_with_base,
    base_toll,
    discovery_cost,
    jackpot_reward_with_base,
    realized_return,
    silence_credit_with_coeffs,
)
from metrics import MetricsTracker
from world import World, decode_gpt2_tokens, encode_gpt2_tokens, encode_utf8_parity_tokens


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return v / n


@dataclass
class StepResult:
    stats: dict
    events: list[str]


class EmeraEngine:
    def __init__(self, cfg: EmeraConfig):
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = World.create(cfg, self.rng)
        self.base_identity: BaseIdentity = create_base_identity(cfg, self.rng)
        self.base_latent = self.base_identity.latent.astype(np.float32)
        self.base_vocab_size = int(self.base_latent.shape[0])
        self.decode_token_ids = np.arange(
            min(cfg.world_vocab_size, self.base_vocab_size), dtype=np.int32
        )
        self.decode_latent = self.base_latent[self.decode_token_ids]
        self.gap = GapField(cfg)
        identity_bank = self._build_initial_identity_bank(cfg.initial_super_tokens)
        self.super_tokens: dict[int, SuperToken] = create_initial_population(
            cfg=cfg,
            rng=self.rng,
            start_token_id=self.base_vocab_size,
            birth_step=0,
            identity_bank=identity_bank,
        )
        self.next_token_id = self.base_vocab_size + cfg.initial_super_tokens
        self.coop = CooperationStats(alpha=cfg.ema_alpha)
        self.metrics = MetricsTracker()
        self.step_idx = 0
        self.last_resonance_strength: dict[int, float] = {}
        self.laws = {
            "attempt_cost_base": float(cfg.attempt_cost_base),
            "jackpot_base": float(cfg.jackpot_base),
            "silence_log_coeff": float(cfg.silence_log_coeff),
            "silence_exp_coeff": float(cfg.silence_exp_coeff),
            "ambient_dissipation": float(cfg.ambient_dissipation),
            "spawn_cost": float(cfg.spawn_cost),
            "mint_delta": float(cfg.mint_delta),
            "pareto_alpha": float(cfg.pareto_alpha_init),
        }
        active0 = float(len(self.super_tokens))
        self.law_ema = {
            "active": active0,
            "match_rate": 0.0,
            "proposal_pressure": 0.0,
            "birth_rate": 0.0,
            "death_rate": 0.0,
        }
        self.law_drive_ema = {
            "collapse_drive": 0.0,
            "overfire": 0.0,
            "underfire": 0.0,
            "imbalance": 0.0,
            "err_active": 0.0,
            "err_match": 0.0,
        }
        self.season_wave_ema = 0.0
        self.energy_reservoir = float(
            np.clip(cfg.energy_reservoir_init, 0.0, cfg.energy_reservoir_cap)
        )
        self.total_energy_ref = self._total_energy()

    def _frontier_latent(self) -> np.ndarray:
        tid = int(np.clip(self.world.peek(1)[0], 0, self.base_latent.shape[0] - 1))
        return self.base_latent[tid]

    def _sample_identity_bytes_from_world(self, length: int) -> np.ndarray:
        l = max(1, int(length))
        if self.world.length <= 0:
            return self.rng.integers(0, 256, size=(l,), dtype=np.int32)
        start = int(self.rng.integers(0, self.world.length))
        idx = (start + np.arange(l, dtype=np.int64)) % self.world.length
        vals = self.world.tokens[idx].astype(np.int32)
        if self.cfg.token_space == "byte_parity":
            vals = (vals & 0xFF).astype(np.int32)
        return vals

    def _build_initial_identity_bank(self, count: int) -> list[np.ndarray]:
        bank: list[np.ndarray] = []
        lmax = max(int(self.cfg.proposal_lmax), 1)
        for _ in range(max(0, int(count))):
            # Bias toward short byte combos (2-4) while still allowing 1-byte specialists.
            if lmax == 1:
                l = 1
            else:
                r = float(self.rng.random())
                if r < 0.15:
                    l = 1
                elif r < 0.55:
                    l = min(2, lmax)
                elif r < 0.85:
                    l = min(3, lmax)
                else:
                    l = lmax
            bank.append(self._sample_identity_bytes_from_world(l))
        return bank

    def _identity_symbol_max(self) -> int:
        if self.cfg.token_space == "byte_parity":
            return 255
        return max(int(self.base_vocab_size) - 1, 0)

    def _sample_gap_genome_fragment(self, preferred_emitters: set[int] | None = None) -> np.ndarray | None:
        lens = np.asarray(self.gap.genome_len, dtype=np.int32)
        valid = lens > 0
        if not np.any(valid):
            return None

        if preferred_emitters:
            emit = np.asarray(self.gap.emitter_id, dtype=np.int64)
            pref_ids = np.fromiter((int(x) for x in preferred_emitters), dtype=np.int64)
            pref_mask = np.isin(emit, pref_ids)
            pick = valid & pref_mask
            if np.any(pick):
                valid = pick

        idx = np.where(valid)[0]
        if idx.size == 0:
            return None
        w = (
            np.asarray(self.gap.energy[idx], dtype=np.float64)
            * np.maximum(np.asarray(self.gap.genome_weight[idx], dtype=np.float64), 1e-6)
        )
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        sw = float(np.sum(w))
        if sw <= 1e-12:
            j = int(self.rng.integers(0, idx.size))
        else:
            probs = w / sw
            j = int(self.rng.choice(idx.size, p=probs))
        slot = int(idx[j])
        n = int(self.gap.genome_len[slot])
        if n <= 0:
            return None
        vmax = self._identity_symbol_max()
        frag = np.asarray(self.gap.genome_fragment[slot, :n], dtype=np.int32)
        return np.clip(frag, 0, vmax).astype(np.int32, copy=True)

    def _mix_gap_genome_fragments(self, fragments: list[np.ndarray]) -> np.ndarray | None:
        seq = [np.asarray(f, dtype=np.int32).reshape(-1) for f in fragments if np.asarray(f).size > 0]
        if not seq:
            return None
        max_len = max(int(self.cfg.proposal_lmax), 1)
        vmax = self._identity_symbol_max()
        if len(seq) == 1:
            out = seq[0].copy()
        else:
            a = seq[0]
            b = seq[1]
            mode = int(self.rng.integers(0, 4))
            if mode == 0:
                na = max(1, a.size // 2)
                nb = max(1, b.size - b.size // 2)
                out = np.concatenate([a[:na], b[-nb:]], axis=0)
            elif mode == 1:
                n = max(1, min(max(a.size, b.size), max_len))
                out = np.empty((n,), dtype=np.int32)
                for i in range(n):
                    ai = min(i, a.size - 1)
                    bi = min(i, b.size - 1)
                    out[i] = int(a[ai]) if (i % 2 == 0) else int(b[bi])
            elif mode == 2:
                out = np.concatenate([a, b], axis=0)
            else:
                out = np.concatenate([b, a], axis=0)
            if out.size > max_len:
                start = int(self.rng.integers(0, out.size - max_len + 1))
                out = out[start : start + max_len]
        if out.size == 0:
            return None
        out = np.clip(out, 0, vmax).astype(np.int32, copy=False)
        if out.size > 0 and self.rng.random() < 0.10:
            k = int(self.rng.integers(0, out.size))
            out = out.copy()
            out[k] = int(self.rng.integers(0, vmax + 1))
        if out.size < max_len and self.rng.random() < 0.08:
            out = np.concatenate([out, np.asarray([int(self.rng.integers(0, vmax + 1))], dtype=np.int32)], axis=0)
        return out[:max_len].astype(np.int32, copy=False)

    def _gap_identity_override(self, preferred_emitters: set[int]) -> np.ndarray | None:
        frags: list[np.ndarray] = []
        primary = self._sample_gap_genome_fragment(preferred_emitters if preferred_emitters else None)
        if primary is not None:
            frags.append(primary)
        if len(preferred_emitters) >= 2:
            secondary = self._sample_gap_genome_fragment(preferred_emitters)
            if secondary is not None:
                frags.append(secondary)
        if len(frags) < 2:
            ambient = self._sample_gap_genome_fragment(None)
            if ambient is not None:
                frags.append(ambient)
        return self._mix_gap_genome_fragments(frags)

    def _sample_gap_ifs_fragment(self, preferred_emitters: set[int] | None = None) -> np.ndarray | None:
        w_base = np.asarray(self.gap.ifs_weight, dtype=np.float32)
        valid = w_base > 1e-9
        if not np.any(valid):
            return None

        if preferred_emitters:
            emit = np.asarray(self.gap.emitter_id, dtype=np.int64)
            pref_ids = np.fromiter((int(x) for x in preferred_emitters), dtype=np.int64)
            pref_mask = np.isin(emit, pref_ids)
            pick = valid & pref_mask
            if np.any(pick):
                valid = pick

        idx = np.where(valid)[0]
        if idx.size == 0:
            return None
        w = (
            np.asarray(self.gap.energy[idx], dtype=np.float64)
            * np.maximum(np.asarray(self.gap.ifs_weight[idx], dtype=np.float64), 1e-6)
        )
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        sw = float(np.sum(w))
        if sw <= 1e-12:
            j = int(self.rng.integers(0, idx.size))
        else:
            probs = w / sw
            j = int(self.rng.choice(idx.size, p=probs))
        slot = int(idx[j])
        return np.asarray(self.gap.ifs_fragment[slot], dtype=np.float32).copy()

    def _mix_gap_ifs_fragments(self, fragments: list[np.ndarray]) -> np.ndarray | None:
        seq = [np.asarray(f, dtype=np.float32).reshape(int(self.cfg.num_ifs), 2, 3) for f in fragments if np.asarray(f).size > 0]
        if not seq:
            return None
        if len(seq) == 1:
            out = seq[0].copy()
        else:
            a = seq[0]
            b = seq[1]
            mode = int(self.rng.integers(0, 3))
            if mode == 0:
                alpha = float(self.rng.uniform(0.30, 0.70))
                out = (alpha * a + (1.0 - alpha) * b).astype(np.float32)
            elif mode == 1:
                choose_b = self.rng.random((int(self.cfg.num_ifs), 1, 1)) < 0.5
                out = np.where(choose_b, b, a).astype(np.float32)
            else:
                out = np.mean(np.stack(seq[: min(3, len(seq))], axis=0), axis=0).astype(np.float32)
        out += self.rng.normal(
            0.0,
            float(self.cfg.ifs_mutation_scale) * 0.35,
            size=out.shape,
        ).astype(np.float32)
        return np.clip(out, -4.0, 4.0).astype(np.float32)

    def _gap_ifs_override(self, preferred_emitters: set[int]) -> np.ndarray | None:
        frags: list[np.ndarray] = []
        primary = self._sample_gap_ifs_fragment(preferred_emitters if preferred_emitters else None)
        if primary is not None:
            frags.append(primary)
        if len(preferred_emitters) >= 2:
            secondary = self._sample_gap_ifs_fragment(preferred_emitters)
            if secondary is not None:
                frags.append(secondary)
        if len(frags) < 2:
            ambient = self._sample_gap_ifs_fragment(None)
            if ambient is not None:
                frags.append(ambient)
        return self._mix_gap_ifs_fragments(frags)

    def _gap_to_latent(self, gap_vec: np.ndarray) -> np.ndarray:
        out = np.zeros((self.cfg.d_latent,), dtype=np.float32)
        out[: self.cfg.gap_dim] = gap_vec
        out += 0.12 * self._frontier_latent()
        return out.astype(np.float32)

    def _identity_tokens_at_cursor(self, identity_bytes: np.ndarray) -> np.ndarray:
        return self._identity_suffix_tokens_at_cursor(identity_bytes, offset=0)

    def _identity_suffix_tokens_at_cursor(self, identity_bytes: np.ndarray, offset: int) -> np.ndarray:
        b = np.asarray(identity_bytes, dtype=np.int32).reshape(-1)
        if b.size == 0:
            return np.zeros((0,), dtype=np.int32)
        start = int(np.clip(offset, 0, b.size - 1))
        if self.cfg.token_space == "byte_parity":
            suffix = np.clip(b[start:], 0, 255).astype(np.int32)
        else:
            suffix = np.clip(b[start:], 0, max(self.base_vocab_size - 1, 0)).astype(np.int32)
        if suffix.size == 0:
            return np.zeros((0,), dtype=np.int32)
        if self.cfg.token_space == "byte_parity":
            frontier_tid = int(self.world.peek(1)[0])
            p0 = (frontier_tid // 256) & 1
            parity = (p0 + np.arange(suffix.size, dtype=np.int32)) & 1
            return (suffix + 256 * parity).astype(np.int32)
        return suffix.astype(np.int32)

    def _identity_frontier_offsets(self, identity_bytes: np.ndarray) -> np.ndarray:
        b = np.asarray(identity_bytes, dtype=np.int32).reshape(-1)
        if b.size == 0:
            return np.zeros((0,), dtype=np.int32)
        if self.cfg.token_space == "byte_parity":
            frontier_byte = int(self.world.peek(1)[0]) & 0xFF
            return np.where(np.clip(b, 0, 255) == frontier_byte)[0].astype(np.int32)
        frontier_tid = int(np.clip(self.world.peek(1)[0], 0, self.base_vocab_size - 1))
        return np.where(np.clip(b, 0, self.base_vocab_size - 1) == frontier_tid)[0].astype(np.int32)

    def _identity_contains_frontier_byte(self, identity_bytes: np.ndarray) -> bool:
        return bool(self._identity_frontier_offsets(identity_bytes).size > 0)

    def _best_identity_alignment(self, identity_bytes: np.ndarray) -> tuple[np.ndarray, int, float, bool]:
        offsets = self._identity_frontier_offsets(identity_bytes)
        if offsets.size == 0:
            return np.zeros((0,), dtype=np.int32), 0, 0.0, False
        best_ids = np.zeros((0,), dtype=np.int32)
        best_key = (-1, -1.0, -1, 0)
        best_m = 0
        best_direct = 0.0
        for off in offsets.tolist():
            ids = self._identity_suffix_tokens_at_cursor(identity_bytes, offset=int(off))
            if ids.size == 0:
                continue
            m = int(self.world.match_prefix_len(ids.tolist()))
            direct = float(m) / float(max(ids.size, 1))
            key = (m, direct, int(ids.size), -int(off))
            if key > best_key:
                best_key = key
                best_ids = ids
                best_m = m
                best_direct = direct
        if best_ids.size == 0:
            return np.zeros((0,), dtype=np.int32), 0, 0.0, False
        return best_ids, best_m, best_direct, True

    def _identity_prefix_fraction(self, identity_bytes: np.ndarray) -> float:
        _, _, direct, found = self._best_identity_alignment(identity_bytes)
        if not found:
            return 0.0
        return float(direct)

    def _decode_from_identity(self, identity_bytes: np.ndarray, resonance_strength: float) -> tuple[np.ndarray, float, float]:
        ids, m, direct, found = self._best_identity_alignment(identity_bytes)
        if ids.size == 0 or not found:
            # Fallback for contrastive probes: decode full identity from offset 0 even
            # when it does not contain the current frontier byte.
            ids = self._identity_tokens_at_cursor(identity_bytes)
            if ids.size == 0:
                return ids, 0.0, 0.0
            m = int(self.world.match_prefix_len(ids.tolist()))
            direct = float(m) / float(max(ids.size, 1))
            frontier_match = 1.0 if int(ids[0]) == int(self.world.peek(1)[0]) else 0.0
        else:
            frontier_match = 1.0
        rs = float(np.clip(resonance_strength, 0.0, 1.0))
        conf = float(np.clip(0.05 + 0.80 * direct + 0.10 * frontier_match + 0.05 * rs, 0.0, 1.0))
        if m == int(ids.size):
            conf = max(conf, 0.90)
        quality = float(np.clip(conf * (0.35 + 0.65 * direct), 0.0, 1.5))
        return ids, conf, quality

    def _attempt_cost(self, token: SuperToken) -> float:
        return attempt_cost_with_base(
            token=token,
            cfg=self.cfg,
            base_cost=float(self.laws["attempt_cost_base"]),
        )

    def _silence_credit(self, token: SuperToken) -> float:
        return silence_credit_with_coeffs(
            token=token,
            cfg=self.cfg,
            log_coeff=float(self.laws["silence_log_coeff"]),
            exp_coeff=float(self.laws["silence_exp_coeff"]),
        )

    def _jackpot(
        self,
        match_len: int,
        proposal_len: int,
        quality: float,
        rarity: float,
        inactivity_steps: int,
    ) -> float:
        return jackpot_reward_with_base(
            cfg=self.cfg,
            jackpot_base_value=float(self.laws["jackpot_base"]),
            match_len=match_len,
            proposal_len=proposal_len,
            quality=quality,
            rarity=rarity,
            inactivity_steps=inactivity_steps,
        )

    def _total_energy(self) -> float:
        tok = 0.0
        for t in self.super_tokens.values():
            if np.isfinite(t.energy):
                tok += max(float(t.energy), 0.0)
        gap_e = float(np.sum(np.clip(self.gap.energy, 0.0, None)))
        return float(tok + self.energy_reservoir + gap_e)

    def _reservoir_add(self, amount: float) -> None:
        if not self.cfg.strict_energy_budget:
            return
        if amount <= 0.0:
            return
        self.energy_reservoir = float(
            np.clip(
                self.energy_reservoir + float(amount),
                0.0,
                self.cfg.energy_reservoir_cap,
            )
        )

    def _reservoir_take(self, requested: float) -> float:
        req = max(float(requested), 0.0)
        if req <= 0.0:
            return 0.0
        if not self.cfg.strict_energy_budget:
            return req
        paid = min(req, self.energy_reservoir)
        self.energy_reservoir -= paid
        return float(paid)

    def _drain_token_energy(self, token: SuperToken, requested: float) -> float:
        req = max(float(requested), 0.0)
        if req <= 0.0:
            return 0.0
        current = float(token.energy) if np.isfinite(token.energy) else 0.0
        current = max(current, 0.0)
        drained = min(current, req)
        token.energy = current - drained
        self._reservoir_add(drained)
        return float(drained)

    def _drain_token_to_gap(self, token: SuperToken, requested: float) -> float:
        req = max(float(requested), 0.0)
        if req <= 0.0:
            return 0.0
        current = float(token.energy) if np.isfinite(token.energy) else 0.0
        current = max(current, 0.0)
        drained = min(current, req)
        token.energy = current - drained
        return float(drained)

    def _credit_token_energy(self, token: SuperToken, requested: float) -> float:
        paid = self._reservoir_take(requested)
        token.energy += paid
        return float(paid)

    def _update_law_emas(self, match_flag: float, proposal_pressure: float, births: int, deaths: int) -> None:
        d = float(np.clip(self.cfg.adaptation_ema_decay, 0.0, 0.9999))
        one = 1.0 - d
        active = float(max(len(self.super_tokens), 1))
        birth_rate = float(births) / active
        death_rate = float(deaths) / active
        self.law_ema["active"] = d * self.law_ema["active"] + one * float(len(self.super_tokens))
        self.law_ema["match_rate"] = d * self.law_ema["match_rate"] + one * float(match_flag)
        self.law_ema["proposal_pressure"] = d * self.law_ema["proposal_pressure"] + one * float(proposal_pressure)
        self.law_ema["birth_rate"] = d * self.law_ema["birth_rate"] + one * birth_rate
        self.law_ema["death_rate"] = d * self.law_ema["death_rate"] + one * death_rate

    def _adapt_natural_laws(self) -> tuple[str, int]:
        if not self.cfg.dynamic_laws:
            return "", 0
        if self.step_idx % self.cfg.law_update_interval != 0:
            return "", 0

        lr = float(max(self.cfg.adaptation_rate, 0.0))
        if lr <= 0.0:
            return "", 0

        ema = self.law_ema
        tgt_active = max(float(self.cfg.target_active_super), 1.0)
        raw_err_active = (tgt_active - ema["active"]) / tgt_active
        raw_err_match = float(self.cfg.target_match_rate) - ema["match_rate"]
        raw_err_pressure = ema["proposal_pressure"] - float(self.cfg.target_proposal_pressure)
        raw_imbalance = (ema["death_rate"] - ema["birth_rate"]) - float(self.cfg.target_birth_death_gap)
        raw_collapse_drive = 0.7 * raw_err_active + 0.6 * raw_err_match + 0.4 * raw_imbalance
        raw_overfire = max(raw_err_pressure, 0.0)
        raw_underfire = max(-raw_err_pressure, 0.0)

        # Low-pass filter adaptation signals so law updates stay responsive but less jumpy.
        signal_decay = float(np.clip(self.cfg.adaptation_signal_decay, 0.0, 0.9999))
        one = 1.0 - signal_decay
        self.law_drive_ema["err_active"] = signal_decay * self.law_drive_ema["err_active"] + one * raw_err_active
        self.law_drive_ema["err_match"] = signal_decay * self.law_drive_ema["err_match"] + one * raw_err_match
        self.law_drive_ema["imbalance"] = signal_decay * self.law_drive_ema["imbalance"] + one * raw_imbalance
        self.law_drive_ema["collapse_drive"] = signal_decay * self.law_drive_ema["collapse_drive"] + one * raw_collapse_drive
        self.law_drive_ema["overfire"] = signal_decay * self.law_drive_ema["overfire"] + one * raw_overfire
        self.law_drive_ema["underfire"] = signal_decay * self.law_drive_ema["underfire"] + one * raw_underfire

        err_active = float(self.law_drive_ema["err_active"])
        err_match = float(self.law_drive_ema["err_match"])
        imbalance = float(self.law_drive_ema["imbalance"])
        collapse_drive = float(self.law_drive_ema["collapse_drive"])
        overfire = max(float(self.law_drive_ema["overfire"]), 0.0)
        underfire = max(float(self.law_drive_ema["underfire"]), 0.0)

        def upd_mul(key: str, drive: float, lo: float, hi: float) -> None:
            val = float(self.laws[key]) * float(np.exp(lr * drive))
            self.laws[key] = float(np.clip(val, lo, hi))

        upd_mul(
            "attempt_cost_base",
            0.8 * overfire - 0.8 * collapse_drive - 0.7 * underfire,
            self.cfg.attempt_cost_min,
            self.cfg.attempt_cost_max,
        )
        upd_mul(
            "jackpot_base",
            1.0 * collapse_drive + 0.8 * underfire - 0.3 * overfire,
            self.cfg.jackpot_base_min,
            self.cfg.jackpot_base_max,
        )
        if not self.cfg.conserve_total_energy:
            upd_mul(
                "silence_log_coeff",
                0.7 * collapse_drive - 0.3 * overfire - 0.5 * underfire,
                self.cfg.silence_log_min,
                self.cfg.silence_log_max,
            )
            upd_mul(
                "silence_exp_coeff",
                0.9 * collapse_drive - 0.3 * overfire - 0.6 * underfire,
                self.cfg.silence_exp_min,
                self.cfg.silence_exp_max,
            )
        upd_mul("ambient_dissipation", 0.5 * overfire - 0.8 * collapse_drive, self.cfg.ambient_dissipation_min, self.cfg.ambient_dissipation_max)
        upd_mul("spawn_cost", -0.8 * imbalance - 0.4 * collapse_drive, self.cfg.spawn_cost_min, self.cfg.spawn_cost_max)

        mint_delta = float(self.laws["mint_delta"]) + lr * (-0.25 * collapse_drive - 0.15 * imbalance)
        self.laws["mint_delta"] = float(np.clip(mint_delta, self.cfg.mint_delta_min, self.cfg.mint_delta_max))

        pareto_alpha = float(self.laws["pareto_alpha"]) - lr * (imbalance + 0.4 * err_active + 0.25 * err_match)
        self.laws["pareto_alpha"] = float(np.clip(pareto_alpha, self.cfg.pareto_alpha_min, self.cfg.pareto_alpha_max))

        season_note, seasonal_births = self._apply_seasonal_forcing()
        law_text = (
            "laws "
            f"attempt={self.laws['attempt_cost_base']:.3f} "
            f"jackpot={self.laws['jackpot_base']:.3f} "
            f"sil_log={self.laws['silence_log_coeff']:.4f} "
            f"sil_exp={self.laws['silence_exp_coeff']:.4f} "
            f"diss={self.laws['ambient_dissipation']:.4f} "
            f"spawn={self.laws['spawn_cost']:.3f} "
            f"mint_delta={self.laws['mint_delta']:.3f} "
            f"pareto_a={self.laws['pareto_alpha']:.3f}"
        )
        if season_note:
            law_text = f"{law_text} | {season_note}"
        return law_text, seasonal_births

    def _spawn_spores(self, count: int) -> int:
        born = 0
        for _ in range(max(0, int(count))):
            e = self._reservoir_take(self.cfg.season_revival_energy)
            if e <= 1e-8:
                break
            tid = self.next_token_id
            token = make_initial_super_token(self.cfg, self.rng, tid, birth_step=self.step_idx)
            token.energy = float(min(e, self.cfg.token_energy_cap))
            self.super_tokens[tid] = token
            self.next_token_id += 1
            born += 1
        return born

    def _apply_seasonal_forcing(self) -> tuple[str, int]:
        if not self.cfg.seasons_enabled:
            return "", 0

        phase = 2.0 * np.pi * (float(self.step_idx % self.cfg.season_period) / float(self.cfg.season_period))
        raw_wave = float(np.sin(phase))
        season_decay = float(np.clip(self.cfg.season_wave_decay, 0.0, 0.9999))
        self.season_wave_ema = season_decay * self.season_wave_ema + (1.0 - season_decay) * raw_wave
        wave = float(np.clip(self.season_wave_ema, -1.0, 1.0))
        renewal = max(wave, 0.0)
        austerity = max(-wave, 0.0)
        s = float(self.cfg.season_strength)

        def upd_mul(key: str, drive: float, lo: float, hi: float) -> None:
            val = float(self.laws[key]) * float(np.exp(s * drive))
            self.laws[key] = float(np.clip(val, lo, hi))

        upd_mul("attempt_cost_base", 0.50 * austerity - 0.50 * renewal, self.cfg.attempt_cost_min, self.cfg.attempt_cost_max)
        upd_mul("jackpot_base", 0.60 * renewal - 0.20 * austerity, self.cfg.jackpot_base_min, self.cfg.jackpot_base_max)
        upd_mul("ambient_dissipation", 0.55 * austerity - 0.35 * renewal, self.cfg.ambient_dissipation_min, self.cfg.ambient_dissipation_max)
        upd_mul("spawn_cost", 0.65 * austerity - 0.85 * renewal, self.cfg.spawn_cost_min, self.cfg.spawn_cost_max)

        mint_delta = float(self.laws["mint_delta"]) + s * (0.030 * austerity - 0.040 * renewal)
        self.laws["mint_delta"] = float(np.clip(mint_delta, self.cfg.mint_delta_min, self.cfg.mint_delta_max))

        spores_born = 0
        if renewal > 0.35 and self.cfg.season_revival_spores > 0:
            target_active = max(int(round(float(self.cfg.target_active_super))), 1)
            active_now = int(len(self.super_tokens))
            deficit = max(target_active - active_now, 0)
            if deficit > 0:
                wave_scale = 0.5 + 1.5 * renewal
                pulse = int(np.ceil(float(self.cfg.season_revival_spores) * wave_scale))
                quota = int(max(1, min(deficit, pulse)))
                spores_born = self._spawn_spores(quota)

        if spores_born > 0:
            return f"season wave={wave:+.2f} renewal spores={spores_born}", spores_born
        return f"season wave={wave:+.2f}", 0

    def _decode_from_state(
        self,
        state: np.ndarray,
        drift: np.ndarray,
        length_bias: float,
        resonance_strength: float,
    ) -> tuple[np.ndarray, float, float]:
        x = float(length_bias) + 2.2 * float(resonance_strength)
        frac = 1.0 / (1.0 + np.exp(-x))
        length = 1 + int(np.floor(frac * float(self.cfg.proposal_lmax - 1)))
        length = int(np.clip(length, 1, self.cfg.proposal_lmax))
        ids = np.zeros((length,), dtype=np.int32)
        confs = np.zeros((length,), dtype=np.float32)
        for k in range(length):
            query = _normalize(state + float(k) * drift)
            scores = self.decode_latent @ query
            top2 = np.argpartition(scores, -2)[-2:]
            if scores[top2[0]] >= scores[top2[1]]:
                i1, i2 = int(top2[0]), int(top2[1])
            else:
                i1, i2 = int(top2[1]), int(top2[0])
            margin = float(scores[i1] - scores[i2])
            conf = 1.0 / (1.0 + np.exp(-self.cfg.confidence_scale * margin))
            ids[k] = int(self.decode_token_ids[i1])
            confs[k] = float(conf)
        confidence = float(np.mean(confs))
        quality = float(confidence * (0.5 + 0.5 * np.clip(resonance_strength, 0.0, 1.0)))
        return ids, confidence, quality

    def _raw_decode_for_token(self, token: SuperToken, resonance_strength: float) -> tuple[np.ndarray, float, float]:
        return self._decode_from_identity(token.identity_bytes, resonance_strength)

    def _proposal_for_token(self, token: SuperToken, resonance_strength: float) -> Optional[Proposal]:
        ids, conf, quality = self._raw_decode_for_token(token, resonance_strength)
        if ids.size <= 0:
            return None
        bet = self._proposal_bet(token, conf)
        if bet <= 0.0:
            return None
        rarity = self.world.rarity_of_sequence(ids.tolist())
        jack_est = self._jackpot(
            match_len=len(ids),
            proposal_len=len(ids),
            quality=quality,
            rarity=rarity,
            inactivity_steps=token.inactivity_steps,
        )
        exp_ret = conf * jack_est - bet
        if self.cfg.obligatory_proposals:
            return Proposal(
                token_id=token.token_id,
                tokens=ids,
                confidence=conf,
                quality=quality,
                expected_return=float(exp_ret),
                bet=float(bet),
            )
        if conf < token.activation_threshold:
            return None
        pressure_deficit = max(
            float(self.cfg.target_proposal_pressure) - float(self.law_ema["proposal_pressure"]),
            0.0,
        )
        explore_prob = float(np.clip(0.05 + 0.45 * pressure_deficit, 0.05, 0.55))
        if exp_ret <= 0.0 and self.rng.random() > explore_prob:
            return None
        return Proposal(
            token_id=token.token_id,
            tokens=ids,
            confidence=conf,
            quality=quality,
            expected_return=float(exp_ret),
            bet=float(bet),
        )

    def _proposal_bet(self, token: SuperToken, confidence: float) -> float:
        unit = max(self._attempt_cost(token), 0.0)
        energy = max(float(token.energy), 0.0)
        cap = float(self.cfg.proposal_bet_max_energy_frac) * energy
        if cap <= 0.0:
            return 0.0
        x = float(self.cfg.proposal_bet_conf_gain) * (float(confidence) - float(token.activation_threshold))
        risk = 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
        floor = float(np.clip(self.cfg.proposal_bet_floor_frac, 0.0, 1.0))
        target = unit * float(self.cfg.proposal_bet_unit_scale) * (floor + (1.0 - floor) * float(risk))
        target = max(target, float(self.cfg.proposal_min_bet))
        return float(min(target, cap))

    def _gate_proposals_to_frontier(self, proposals: list[Proposal]) -> tuple[list[Proposal], int, int]:
        if not proposals:
            return [], 0, 0

        frontier_tid = int(self.world.peek(1)[0])
        frontier_props: list[Proposal] = []
        other_props: list[Proposal] = []
        for p in proposals:
            if p.tokens.size > 0 and int(p.tokens[0]) == frontier_tid:
                frontier_props.append(p)
            else:
                other_props.append(p)

        if frontier_props:
            n = min(len(other_props), int(self.cfg.proposal_frontier_contrast))
            if n > 0:
                idx = np.asarray(self.rng.choice(len(other_props), size=n, replace=False), dtype=np.int64)
                sampled = [other_props[int(i)] for i in idx.tolist()]
            else:
                sampled = []
            gated = frontier_props + sampled
            return gated, len(frontier_props), len(gated)

        return [], 0, 0

    def _spawn_frontier_specialist(self, frontier_tid: int) -> Optional[SuperToken]:
        e = self._reservoir_take(float(self.cfg.frontier_rescue_energy))
        if e <= 1e-8:
            return None

        tid = self.next_token_id
        l = int(self.rng.integers(1, self.cfg.proposal_lmax + 1))
        ident = self.world.peek(l).astype(np.int32)
        if self.cfg.token_space == "byte_parity":
            ident = (ident & 0xFF).astype(np.int32)
        token = make_initial_super_token(
            self.cfg,
            self.rng,
            tid,
            birth_step=self.step_idx,
            identity_bytes=ident,
        )
        target = self.base_latent[int(np.clip(frontier_tid, 0, self.base_latent.shape[0] - 1))].astype(np.float32)
        noise = self.rng.normal(0.0, float(self.cfg.frontier_rescue_noise), size=target.shape).astype(np.float32)

        token.state_vec = _normalize((target + noise).astype(np.float32)).astype(np.float32)
        token.signature = _normalize(token.state_vec[: self.cfg.gap_dim]).astype(np.float32)
        token.phase = float(np.arctan2(token.state_vec[1], token.state_vec[0]))
        token.proposal_drift = _normalize(
            0.70 * token.proposal_drift + 0.30 * target
        ).astype(np.float32)
        token.activation_threshold = float(min(token.activation_threshold, 0.35))
        token.proposal_length_bias = float(min(token.proposal_length_bias, -0.80))
        token.energy = float(min(e, self.cfg.token_energy_cap))

        self.super_tokens[tid] = token
        self.next_token_id += 1
        return token

    def _proposal_for_pair_fusion(self, a: SuperToken, b: SuperToken) -> tuple[np.ndarray, float, float]:
        rs = 0.5 * (
            self.last_resonance_strength.get(a.token_id, 0.0)
            + self.last_resonance_strength.get(b.token_id, 0.0)
        )
        ia = np.asarray(a.identity_bytes, dtype=np.int32).reshape(-1)
        ib = np.asarray(b.identity_bytes, dtype=np.int32).reshape(-1)
        if ia.size == 0:
            ia = np.asarray([32], dtype=np.int32)
        if ib.size == 0:
            ib = np.asarray([32], dtype=np.int32)
        na = max(1, ia.size // 2)
        nb = max(1, ib.size - ib.size // 2)
        fused = np.concatenate([ia[:na], ib[-nb:]], axis=0)[: max(1, self.cfg.proposal_lmax)]
        ids_f, conf_f, q_f = self._decode_from_identity(fused, rs)
        ids_a, conf_a, q_a = self._decode_from_identity(ia, self.last_resonance_strength.get(a.token_id, 0.0))
        ids_b, conf_b, q_b = self._decode_from_identity(ib, self.last_resonance_strength.get(b.token_id, 0.0))

        candidates = [(ids_f, conf_f, q_f), (ids_a, conf_a, q_a), (ids_b, conf_b, q_b)]
        ids, conf, quality = max(candidates, key=lambda x: x[1] if x[0].size > 0 else -1.0)

        align = float(np.clip(np.dot(_normalize(a.state_vec), _normalize(b.state_vec)), 0.0, 1.0))
        conf = float(np.clip(conf + 0.18 * align, 0.0, 1.0))
        quality = float(np.clip(quality + 0.22 * align, 0.0, 1.5))
        return ids, conf, quality

    def _evaluate_proposal_return(
        self,
        proposal_tokens: np.ndarray,
        confidence: float,
        quality: float,
        inactivity_steps: int,
        attempt_total: float,
    ) -> float:
        p_len = int(proposal_tokens.size)
        if p_len <= 0:
            d = discovery_cost(1, self.cfg)
            b = base_toll(1, self.cfg)
            return realized_return(0.0, attempt_total, d, b)

        m = self.world.match_prefix_len(proposal_tokens.tolist())
        adv = max(1, m)
        unmatched = 0 if m == p_len else max(1, p_len - m)
        d = 0.0 if m == p_len else discovery_cost(unmatched, self.cfg)
        rarity = self.world.rarity_of_sequence(proposal_tokens[: max(m, 1)].tolist())
        q = float(confidence * (float(m) / max(float(p_len), 1.0)))
        j = self._jackpot(
            match_len=m,
            proposal_len=p_len,
            quality=q,
            rarity=rarity,
            inactivity_steps=inactivity_steps,
        )
        b = base_toll(adv, self.cfg)
        return realized_return(j, attempt_total, d, b)

    def _proposal_realized_components(self, proposal: Proposal) -> dict[str, float | int]:
        p_len = int(proposal.tokens.size)
        if p_len <= 0:
            adv = 1
            disc = discovery_cost(1, self.cfg)
            j = 0.0
            score = realized_return(j, float(proposal.bet), disc, base_toll(adv, self.cfg))
            return {
                "match_len": 0,
                "proposal_len": 0,
                "advance_len": adv,
                "discovery_cost": float(disc),
                "jackpot": float(j),
                "score": float(score),
            }

        m = self.world.match_prefix_len(proposal.tokens.tolist())
        adv = max(1, m)
        full = m == p_len
        unmatched = 0 if full else max(1, p_len - m)
        disc = 0.0 if full else discovery_cost(unmatched, self.cfg)
        rarity = self.world.rarity_of_sequence(proposal.tokens[: max(m, 1)].tolist())
        q = float(proposal.quality * (float(m) / max(float(p_len), 1.0)))
        tok = self.super_tokens.get(proposal.token_id)
        inactivity = tok.inactivity_steps if tok is not None else 0
        j = self._jackpot(
            match_len=m,
            proposal_len=p_len,
            quality=q,
            rarity=rarity,
            inactivity_steps=inactivity,
        )
        score = realized_return(j, float(proposal.bet), disc, base_toll(adv, self.cfg))
        return {
            "match_len": int(m),
            "proposal_len": int(p_len),
            "advance_len": int(adv),
            "discovery_cost": float(disc),
            "jackpot": float(j),
            "score": float(score),
        }

    def _ablation_returns(self, a: SuperToken, b: SuperToken) -> tuple[float, float, float]:
        ids_a, conf_a, q_a = self._raw_decode_for_token(a, self.last_resonance_strength.get(a.token_id, 0.0))
        ids_b, conf_b, q_b = self._raw_decode_for_token(b, self.last_resonance_strength.get(b.token_id, 0.0))
        ids_ab, conf_ab, q_ab = self._proposal_for_pair_fusion(a, b)
        r_a = self._evaluate_proposal_return(
            proposal_tokens=ids_a,
            confidence=conf_a,
            quality=q_a,
            inactivity_steps=a.inactivity_steps,
            attempt_total=self._attempt_cost(a),
        )
        r_b = self._evaluate_proposal_return(
            proposal_tokens=ids_b,
            confidence=conf_b,
            quality=q_b,
            inactivity_steps=b.inactivity_steps,
            attempt_total=self._attempt_cost(b),
        )
        r_ab = self._evaluate_proposal_return(
            proposal_tokens=ids_ab,
            confidence=conf_ab,
            quality=q_ab,
            inactivity_steps=max(a.inactivity_steps, b.inactivity_steps),
            attempt_total=max(self._attempt_cost(a), self._attempt_cost(b)),
        )
        return r_ab, r_a, r_b

    def _adapt_winner_from_truth(self, winner: SuperToken, gt_tokens: np.ndarray, match_len: int) -> None:
        if gt_tokens.size == 0:
            return
        if match_len > 0:
            target = _normalize(np.mean(self.base_latent[gt_tokens[:match_len]], axis=0))
            winner.state_vec = _normalize((1.0 - 0.06) * winner.state_vec + 0.06 * target).astype(np.float32)
        if match_len < gt_tokens.size:
            miss = self.base_latent[int(gt_tokens[match_len])]
            winner.state_vec = _normalize((1.0 - 0.18) * winner.state_vec + 0.18 * miss).astype(np.float32)
            winner.proposal_drift = _normalize(
                0.92 * winner.proposal_drift + 0.08 * (miss - winner.state_vec)
            ).astype(np.float32)
        winner.signature = _normalize(
            0.95 * winner.signature + 0.05 * winner.state_vec[: self.cfg.gap_dim]
        ).astype(np.float32)
        winner.phase = float(np.arctan2(winner.state_vec[1], winner.state_vec[0]))

    def _apply_jackpot(self, winner: SuperToken, jackpot: float) -> None:
        if jackpot <= 0.0:
            return
        payout = self._reservoir_take(jackpot)
        if payout <= 0.0:
            return
        a = winner.parent_a
        b = winner.parent_b
        parent_a = self.super_tokens.get(a)
        parent_b = self.super_tokens.get(b)
        if parent_a is not None and parent_b is not None:
            winner.energy += self.cfg.child_reward_share * payout
            parent_a.energy += self.cfg.parent_reward_share * payout
            parent_b.energy += self.cfg.parent_reward_share * payout
            return
        winner.energy += payout

    def _self_copy_cycle(self, proposal_evals: list[tuple[Proposal, dict[str, float | int]]]) -> tuple[int, list[str]]:
        cfg = self.cfg
        if not bool(cfg.self_copy_enabled):
            return 0, []
        if int(cfg.self_copy_max_per_step) <= 0:
            return 0, []
        if int(self.step_idx % int(cfg.self_copy_interval)) != 0:
            return 0, []
        if not proposal_evals:
            return 0, []

        copy_cost = float(cfg.self_copy_cost)
        min_energy = float(cfg.self_copy_min_energy)
        min_match_frac = float(cfg.self_copy_min_match_frac)
        min_score = float(cfg.self_copy_min_score)

        births = 0
        events: list[str] = []
        ranked = sorted(
            proposal_evals,
            key=lambda pe: (
                float(pe[1].get("score", 0.0)),
                int(pe[1].get("match_len", 0)),
                float(pe[0].confidence),
            ),
            reverse=True,
        )

        for proposal, evals in ranked:
            if births >= int(cfg.self_copy_max_per_step):
                break
            parent = self.super_tokens.get(int(proposal.token_id))
            if parent is None:
                continue

            plen = max(int(evals.get("proposal_len", 0)), 1)
            mlen = max(int(evals.get("match_len", 0)), 0)
            match_frac = float(mlen) / float(plen)
            score = float(evals.get("score", 0.0))
            if match_frac < min_match_frac or score < min_score:
                continue
            if float(parent.energy) < (min_energy + copy_cost):
                continue

            paid = self._drain_token_energy(parent, copy_cost)
            if paid <= 1e-8:
                continue
            child_energy = self._reservoir_take(paid)
            if child_energy <= 1e-8:
                continue

            gap_ident = self._gap_identity_override({int(parent.token_id)})
            gap_ifs = self._gap_ifs_override({int(parent.token_id)})
            child = mint_child_from_parent(
                cfg=cfg,
                rng=self.rng,
                new_id=self.next_token_id,
                parent=parent,
                spawn_energy=child_energy,
                pareto_alpha=float(self.laws["pareto_alpha"]),
                identity_override=gap_ident,
                ifs_override=gap_ifs,
                birth_step=self.step_idx,
            )
            self.super_tokens[child.token_id] = child
            self.next_token_id += 1
            births += 1
            events.append(
                f"copy {child.token_id} <- {parent.token_id} score={score:.3f} match={mlen}/{plen} "
                f"gap_genome={1 if gap_ident is not None else 0} glen={int(gap_ident.size) if gap_ident is not None else 0} "
                f"gap_ifs={1 if gap_ifs is not None else 0}"
            )

        return births, events

    def _mint_cycle(self) -> tuple[int, list[str]]:
        births = 0
        events: list[str] = []
        spawn_cost = float(self.laws["spawn_cost"])
        mint_delta = float(self.laws["mint_delta"])
        half_spawn = 0.5 * spawn_cost
        for a_id, b_id, syn, hits in self.coop.top_pairs(min_hits=1):
            a = self.super_tokens.get(a_id)
            b = self.super_tokens.get(b_id)
            if a is None or b is None:
                continue
            if a.energy < half_spawn or b.energy < half_spawn:
                continue
            r_ab, r_a, r_b = self._ablation_returns(a, b)
            if r_ab <= max(r_a, r_b) + mint_delta:
                continue

            self._drain_token_energy(a, half_spawn)
            self._drain_token_energy(b, half_spawn)
            child_energy = self._reservoir_take(spawn_cost)
            if child_energy <= 1e-8:
                continue
            gap_ident = self._gap_identity_override({int(a_id), int(b_id)})
            gap_ifs = self._gap_ifs_override({int(a_id), int(b_id)})
            child = mint_child_from_parents(
                cfg=self.cfg,
                rng=self.rng,
                new_id=self.next_token_id,
                parent_a=a,
                parent_b=b,
                spawn_energy=child_energy,
                pareto_alpha=float(self.laws["pareto_alpha"]),
                identity_override=gap_ident,
                ifs_override=gap_ifs,
                birth_step=self.step_idx,
            )
            self.super_tokens[child.token_id] = child
            self.next_token_id += 1
            births += 1
            events.append(
                f"mint {child.token_id} <- ({a_id},{b_id}) syn={syn:.3f} hits={hits} "
                f"ab={r_ab:.3f} a={r_a:.3f} b={r_b:.3f} "
                f"gap_genome={1 if gap_ident is not None else 0} glen={int(gap_ident.size) if gap_ident is not None else 0} "
                f"gap_ifs={1 if gap_ifs is not None else 0}"
            )
        return births, events

    def step(self) -> StepResult:
        self.step_idx += 1
        events: list[str] = []
        cfg = self.cfg
        if not self.cfg.conserve_total_energy and cfg.energy_inflow_per_step > 0.0:
            self._reservoir_add(cfg.energy_inflow_per_step)

        active_rounds: dict[int, int] = {}
        last_strength: dict[int, float] = {}
        prefix_match_frac: dict[int, float] = {}
        for tid, token in self.super_tokens.items():
            if token.energy <= cfg.min_viable_energy:
                continue
            prefix_match_frac[tid] = self._identity_prefix_fraction(token.identity_bytes)

        # K synchronized rounds; writes are immediate and visible in following rounds.
        for round_idx in range(cfg.k_rounds):
            if self.cfg.conserve_total_energy:
                lost = float(np.sum(self.gap.energy) * max(1.0 - self.cfg.gap_decay, 0.0))
                self._reservoir_add(lost)
            self.gap.decay()
            for tid in list(self.super_tokens.keys()):
                token = self.super_tokens.get(tid)
                if token is None or token.energy <= cfg.min_viable_energy:
                    continue
                read = self.gap.read(
                    receiver_signature=token.signature,
                    resonance_width=token.resonance_width,
                    receiver_phase=token.phase,
                    phase_coupling=token.phase_coupling,
                )
                resonance_latent = self._gap_to_latent(read.resonance)
                substeps = max(int(self.cfg.chaos_substeps_per_round), 1)
                vel_sum = np.zeros((2,), dtype=np.float32)
                vel2 = vel_sum
                for _ in range(substeps):
                    vel2 = token.chaos_step(self.rng, resonance_latent)
                    vel_sum += vel2.astype(np.float32)
                vel2 = (vel_sum / float(substeps)).astype(np.float32)
                last_strength[tid] = read.strength

                prefix_score = float(prefix_match_frac.get(tid, 0.0))
                wake_score = 0.85 * prefix_score + 0.15 * float(read.strength)
                activate = wake_score >= token.activation_threshold
                if not activate:
                    continue
                emit, vel, eng = token.emission(cfg, vel2)
                if eng <= cfg.min_viable_energy:
                    continue
                eng_paid = self._drain_token_to_gap(token, eng)
                if eng_paid <= cfg.min_viable_energy:
                    if self.cfg.conserve_total_energy and eng_paid > 0.0:
                        self._reservoir_add(eng_paid)
                    continue
                if self.cfg.conserve_total_energy:
                    overwritten = float(self.gap.energy[self.gap.ptr])
                    if overwritten > 0.0:
                        self._reservoir_add(overwritten)
                self.gap.write(
                    point=emit,
                    velocity=vel,
                    phase=token.phase,
                    omega=token.omega,
                    energy=eng_paid,
                    genome_fragment=token.identity_bytes,
                    genome_weight=float(read.strength),
                    ifs_fragment=token.ifs,
                    ifs_weight=float(read.strength),
                    emitter_id=tid,
                    step_idx=self.step_idx,
                    round_idx=round_idx,
                )
                active_rounds[tid] = active_rounds.get(tid, 0) + 1

        self.last_resonance_strength = last_strength

        active_ids = [tid for tid, cnt in active_rounds.items() if cnt > 0 and tid in self.super_tokens]
        if self.cfg.obligatory_proposals:
            frontier_ids: list[int] = []
            non_frontier_ids: list[int] = []
            for tid, token in self.super_tokens.items():
                if token.energy <= cfg.min_viable_energy:
                    continue
                if self._identity_contains_frontier_byte(token.identity_bytes):
                    frontier_ids.append(int(tid))
                else:
                    non_frontier_ids.append(int(tid))

            if frontier_ids:
                n = min(len(non_frontier_ids), int(cfg.proposal_frontier_contrast))
                if n > 0:
                    ridx = np.asarray(self.rng.choice(len(non_frontier_ids), size=n, replace=False), dtype=np.int64)
                    contrast = [non_frontier_ids[int(i)] for i in ridx.tolist()]
                else:
                    contrast = []
                proposal_ids = frontier_ids + contrast
            else:
                proposal_ids = []
        else:
            proposal_ids = active_ids
        proposals: list[Proposal] = []
        for tid in proposal_ids:
            token = self.super_tokens[tid]
            prop = self._proposal_for_token(token, last_strength.get(tid, 0.0))
            if prop is not None:
                proposals.append(prop)
        proposals_raw_count = len(proposals)
        proposals, frontier_match_count, proposals_gated_count = self._gate_proposals_to_frontier(proposals)
        frontier_rescue_spawned = 0

        proposer_ids = [int(p.token_id) for p in proposals]

        winner: Optional[Proposal] = None
        winner_eval: Optional[dict[str, float | int]] = None
        proposal_evals: list[tuple[Proposal, dict[str, float | int]]] = []
        if proposals:
            for p in proposals:
                e = self._proposal_realized_components(p)
                proposal_evals.append((p, e))
                if winner is None or winner_eval is None:
                    winner = p
                    winner_eval = e
                    continue
                # Selection is environmental: best realized score wins.
                if (
                    float(e["score"]) > float(winner_eval["score"])
                    or (
                        float(e["score"]) == float(winner_eval["score"])
                        and int(e["match_len"]) > int(winner_eval["match_len"])
                    )
                    or (
                        float(e["score"]) == float(winner_eval["score"])
                        and int(e["match_len"]) == int(winner_eval["match_len"])
                        and float(p.confidence) > float(winner.confidence)
                    )
                ):
                    winner = p
                    winner_eval = e

        longest_silent_turn_id = -1
        longest_silent_turn_steps = 0
        longest_silent_turn_conf = 0.0
        longest_silent_turn_match_len = 0
        longest_silent_turn_proposal_len = 0
        longest_silent_turn_score = 0.0
        longest_silent_turn_won = 0
        longest_silent_turn_token_len = 0
        longest_silent_turn_parent_a = -1
        longest_silent_turn_parent_b = -1
        longest_silent_turn_identity_bytes: list[int] = []
        if proposal_evals:
            def _silence_for(prop: Proposal) -> int:
                tok = self.super_tokens.get(int(prop.token_id))
                if tok is None:
                    return -1
                return int(tok.inactivity_steps)

            best_prop, best_eval = max(
                proposal_evals,
                key=lambda pe: (
                    _silence_for(pe[0]),
                    float(pe[0].confidence),
                    int(pe[0].token_id),
                ),
            )
            best_tok = self.super_tokens.get(int(best_prop.token_id))
            longest_silent_turn_id = int(best_prop.token_id)
            longest_silent_turn_steps = max(_silence_for(best_prop), 0)
            longest_silent_turn_conf = float(best_prop.confidence)
            longest_silent_turn_match_len = int(best_eval["match_len"])
            longest_silent_turn_proposal_len = int(best_eval["proposal_len"])
            longest_silent_turn_score = float(best_eval["score"])
            longest_silent_turn_won = int(
                winner is not None and int(winner.token_id) == int(best_prop.token_id)
            )
            if best_tok is not None:
                ib = np.asarray(best_tok.identity_bytes, dtype=np.int32).reshape(-1)
                longest_silent_turn_token_len = int(ib.size)
                longest_silent_turn_parent_a = int(best_tok.parent_a)
                longest_silent_turn_parent_b = int(best_tok.parent_b)
                longest_silent_turn_identity_bytes = [int(x) for x in ib.tolist()]

        attempt_map: Dict[int, float] = {int(p.token_id): max(float(p.bet), 0.0) for p in proposals}
        attempt_total = float(sum(attempt_map.values()))

        if winner is None:
            match_len = 0
            advance_len = 1
            unmatched_len = 1
            disc_cost = discovery_cost(unmatched_len, cfg)
            jackpot = 0.0
            winner_id = None
            winner_conf = 0.0
            proposal_len = 0
            winner_super_token_len = 0
            winner_inactivity_before = 0
        else:
            assert winner_eval is not None
            proposal_len = int(winner_eval["proposal_len"])
            gt_tokens = self.world.peek(proposal_len)
            match_len = int(winner_eval["match_len"])
            advance_len = int(winner_eval["advance_len"])
            disc_cost = float(winner_eval["discovery_cost"])
            jackpot = float(winner_eval["jackpot"])
            winner_id = winner.token_id
            winner_conf = winner.confidence
            winner_token_ref = self.super_tokens.get(winner.token_id)
            if winner_token_ref is not None:
                winner_super_token_len = int(np.asarray(winner_token_ref.identity_bytes, dtype=np.int32).size)
                winner_inactivity_before = int(winner_token_ref.inactivity_steps)
            else:
                winner_super_token_len = 0
                winner_inactivity_before = 0

        base = base_toll(advance_len, cfg)

        # Ambient thermodynamic dissipation.
        for token in self.super_tokens.values():
            diss_rate = float(self.laws["ambient_dissipation"])
            self._drain_token_energy(token, max(token.energy, 0.0) * diss_rate)

        # Baseline metabolism: a small universal tax that recycles into reservoir.
        actual_metabolic_tax = 0.0
        if cfg.metabolic_tax_rate > 0.0:
            for token in self.super_tokens.values():
                actual_metabolic_tax += self._drain_token_energy(
                    token,
                    max(token.energy, 0.0) * float(cfg.metabolic_tax_rate),
                )

        # Attempts cost the proposers.
        actual_attempt_total = 0.0
        for tid, c in attempt_map.items():
            token = self.super_tokens.get(tid)
            if token is not None:
                actual_attempt_total += self._drain_token_energy(token, c)

        # Contrastive settlement over all active proposals:
        # confident correctness gets rewarded, confident errors get penalized.
        actual_contrastive_bonus = 0.0
        actual_contrastive_penalty = 0.0
        if bool(self.cfg.contrastive_enabled):
            for p, e in proposal_evals:
                token = self.super_tokens.get(int(p.token_id))
                if token is None:
                    continue
                plen = max(int(e["proposal_len"]), 1)
                mlen = int(e["match_len"])
                acc = float(np.clip(float(mlen) / float(plen), 0.0, 1.0))
                conf = float(np.clip(p.confidence, 0.0, 1.0))
                bonus_req = float(self.cfg.contrastive_correct_reward) * conf * acc
                wrong = float(1.0 - acc)
                penalty_req = float(self.cfg.contrastive_wrong_penalty) * (conf ** float(self.cfg.contrastive_wrong_exp)) * wrong

                if penalty_req > 0.0:
                    actual_contrastive_penalty += self._drain_token_energy(token, penalty_req)
                if bonus_req > 0.0:
                    actual_contrastive_bonus += self._credit_token_energy(token, bonus_req)

        # Discovery penalty is paid by the winning proposer.
        actual_discovery_cost = 0.0
        if winner is not None and disc_cost > 0.0:
            token = self.super_tokens.get(winner.token_id)
            if token is not None:
                actual_discovery_cost += self._drain_token_energy(token, disc_cost)

        # Travel is never free: every alive token pays base toll each step.
        actual_base_toll = 0.0
        if base > 0.0:
            for token in self.super_tokens.values():
                actual_base_toll += self._drain_token_energy(token, base)

        # Jackpot reward to winner (or winner+parents if child).
        actual_jackpot = 0.0
        if winner is not None and jackpot > 0.0:
            token = self.super_tokens.get(winner.token_id)
            if token is not None:
                before = float(self.energy_reservoir)
                self._apply_jackpot(token, jackpot)
                after = float(self.energy_reservoir)
                if self.cfg.strict_energy_budget:
                    actual_jackpot = max(before - after, 0.0)
                else:
                    actual_jackpot = jackpot
        if winner is not None:
            token = self.super_tokens.get(winner.token_id)
            if token is not None:
                self._adapt_winner_from_truth(token, gt_tokens, match_len)

        r_step = float(
            actual_jackpot
            + actual_contrastive_bonus
            - actual_attempt_total
            - actual_discovery_cost
            - actual_base_toll
            - actual_contrastive_penalty
        )

        if winner is not None and winner_eval is not None:
            token = self.super_tokens.get(winner.token_id)
            if token is not None:
                if match_len > 0:
                    token.max_silent_correct = max(
                        int(token.max_silent_correct),
                        int(winner_inactivity_before),
                    )
                if float(winner_eval["score"]) > 0.0 and float(winner.bet) > 0.0:
                    token.max_paid_bet = max(float(token.max_paid_bet), float(winner.bet))

        proposer_set = set(proposer_ids)
        inactive_ids: list[int] = []
        silence_req: list[float] = []
        for tid, token in list(self.super_tokens.items()):
            if tid in proposer_set:
                token.inactivity_steps = 0
            else:
                token.inactivity_steps += 1
                inactive_ids.append(tid)
                silence_req.append(self._silence_credit(token))
        if inactive_ids and silence_req:
            total_req = float(sum(silence_req))
            if not self.cfg.strict_energy_budget:
                for tid, req in zip(inactive_ids, silence_req):
                    token = self.super_tokens.get(tid)
                    if token is not None:
                        token.energy += req
            elif total_req > 0.0 and self.energy_reservoir > 0.0:
                if self.cfg.conserve_total_energy:
                    step_pool = actual_attempt_total + actual_discovery_cost + actual_base_toll
                    relief_budget = 0.0
                    active_now = len(self.super_tokens)
                    target_active = max(float(self.cfg.target_active_super), 1.0)
                    if (
                        active_now > 0
                        and float(active_now) < float(self.cfg.survivor_relief_active_frac) * target_active
                    ):
                        relief_budget = min(
                            self.energy_reservoir,
                            float(self.cfg.survivor_relief_reservoir_frac) * self.energy_reservoir,
                        )
                    max_pay = min(total_req, self.energy_reservoir, step_pool + relief_budget)
                else:
                    max_pay = min(total_req, self.energy_reservoir)
                factor = min(1.0, max_pay / total_req) if total_req > 0.0 else 0.0
                paid_total = 0.0
                for tid, req in zip(inactive_ids, silence_req):
                    token = self.super_tokens.get(tid)
                    if token is None:
                        continue
                    pay = req * factor
                    if pay < cfg.min_viable_energy:
                        continue
                    token.energy += pay
                    paid_total += pay
                self.energy_reservoir = max(0.0, self.energy_reservoir - paid_total)

        self.world.advance(advance_len)

        # Update cooperation signal from winning context.
        if winner is not None and winner.token_id in self.super_tokens:
            wt = self.super_tokens[winner.token_id]
            contrib = self.gap.contribution_weights(
                receiver_signature=wt.signature,
                resonance_width=wt.resonance_width,
                receiver_phase=wt.phase,
                phase_coupling=wt.phase_coupling,
            )
            if not contrib:
                contrib = {winner.token_id: 1.0}
            self.coop.update(contrib, r_step)

        self_copy_births = 0
        if proposal_evals:
            b_copy, copy_events = self._self_copy_cycle(proposal_evals)
            self_copy_births += b_copy
            events.extend(copy_events)

        births = int(self_copy_births)

        # Keep token numerics bounded to avoid NaN cascades.
        for token in self.super_tokens.values():
            if not np.isfinite(token.energy):
                token.energy = 0.0
            token.energy = float(np.clip(token.energy, 0.0, self.cfg.token_energy_cap))
            if token.energy < cfg.min_viable_energy:
                token.low_energy_steps = int(getattr(token, "low_energy_steps", 0)) + 1
            else:
                token.low_energy_steps = 0
            token.state_vec = np.nan_to_num(token.state_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            token.signature = np.nan_to_num(token.signature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            token.proposal_drift = np.nan_to_num(
                token.proposal_drift, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            if not np.isfinite(token.phase):
                token.phase = 0.0
            if not np.isfinite(token.omega):
                token.omega = float(self.cfg.omega_base)

        dead_ids: list[int] = []
        dead_token_energy = 0.0
        for tid, token in list(self.super_tokens.items()):
            if token.energy <= 0.0:
                dead_ids.append(tid)
                dead_token_energy += max(float(token.energy), 0.0)
                del self.super_tokens[tid]
                continue
            if token.energy < cfg.min_viable_energy and int(getattr(token, "low_energy_steps", 0)) > cfg.survivor_grace_steps:
                dead_ids.append(tid)
                dead_token_energy += max(float(token.energy), 0.0)
                del self.super_tokens[tid]
        if dead_ids:
            dead_arr = np.asarray(dead_ids, dtype=np.int64)
            dead_gap_mask = np.isin(self.gap.emitter_id, dead_arr)
            recycled_gap = float(np.sum(self.gap.energy[dead_gap_mask]))
            if self.cfg.conserve_total_energy:
                recycled = recycled_gap + dead_token_energy
            else:
                recycled = (
                    float(self.cfg.death_recycle_fraction) * recycled_gap
                    + float(self.cfg.death_recycle_flat) * float(len(dead_ids))
                )
            self._reservoir_add(recycled)
            self.gap.purge_emitters(dead_ids)
            if recycled > 0.0:
                events.append(f"decompose +{recycled:.3f}")
        self.coop.prune_dead(self.super_tokens.keys())

        if (self.step_idx % cfg.mint_interval) == 0 and len(self.super_tokens) >= 2:
            b, mint_events = self._mint_cycle()
            births += b
            events.extend(mint_events)

        deaths = len(dead_ids)
        if deaths > 0:
            events.append(f"deaths {deaths}")

        active_ref = max(len(self.super_tokens) + deaths, 1)
        proposal_pressure = float(len(proposer_ids) / float(active_ref))
        match_flag = 1.0 if match_len > 0 else 0.0
        self._update_law_emas(
            match_flag=match_flag,
            proposal_pressure=proposal_pressure,
            births=births,
            deaths=deaths,
        )
        law_event, seasonal_births = self._adapt_natural_laws()
        births += int(seasonal_births)
        if law_event:
            events.append(law_event)

        discovery_advance = max(advance_len - match_len, 0)
        stats = {
            "step": self.step_idx,
            "active_super": len(self.super_tokens),
            "proposers": len(proposer_ids),
            "proposers_raw": int(proposals_raw_count),
            "proposers_frontier": int(frontier_match_count),
            "proposers_gated": int(proposals_gated_count),
            "frontier_rescue_spawned": int(frontier_rescue_spawned),
            "winner_id": -1 if winner_id is None else int(winner_id),
            "winner_conf": float(winner_conf),
            "proposal_len": int(proposal_len),
            "winner_super_token_len": int(winner_super_token_len),
            "match_len": int(match_len),
            "advance_len": int(advance_len),
            "discovery_advance": int(discovery_advance),
            "attempt_total": float(attempt_total),
            "attempt_total_actual": float(actual_attempt_total),
            "attempt_avg": float(attempt_total / max(len(proposer_ids), 1)),
            "contrastive_bonus": float(actual_contrastive_bonus),
            "contrastive_penalty": float(actual_contrastive_penalty),
            "discovery_cost": float(actual_discovery_cost),
            "base_toll": float(actual_base_toll),
            "jackpot": float(actual_jackpot),
            "jackpot_requested": float(jackpot),
            "realized_return": float(r_step),
            "energy_spent": float(
                actual_attempt_total
                + actual_contrastive_penalty
                + actual_discovery_cost
                + actual_base_toll
                + actual_metabolic_tax
            ),
            "metabolic_tax": float(actual_metabolic_tax),
            "births": int(births),
            "self_copy_births": int(self_copy_births),
            "deaths": int(deaths),
            "cursor": int(self.world.cursor),
            "longest_silent_turn_id": int(longest_silent_turn_id),
            "longest_silent_turn_steps": int(longest_silent_turn_steps),
            "longest_silent_turn_conf": float(longest_silent_turn_conf),
            "longest_silent_turn_match_len": int(longest_silent_turn_match_len),
            "longest_silent_turn_proposal_len": int(longest_silent_turn_proposal_len),
            "longest_silent_turn_score": float(longest_silent_turn_score),
            "longest_silent_turn_won": int(longest_silent_turn_won),
            "longest_silent_turn_token_len": int(longest_silent_turn_token_len),
            "longest_silent_turn_parent_a": int(longest_silent_turn_parent_a),
            "longest_silent_turn_parent_b": int(longest_silent_turn_parent_b),
            "longest_silent_turn_identity_bytes": list(longest_silent_turn_identity_bytes),
            "law_attempt_cost_base": float(self.laws["attempt_cost_base"]),
            "law_jackpot_base": float(self.laws["jackpot_base"]),
            "law_spawn_cost": float(self.laws["spawn_cost"]),
            "law_mint_delta": float(self.laws["mint_delta"]),
            "law_pareto_alpha": float(self.laws["pareto_alpha"]),
            "reservoir": float(self.energy_reservoir),
            "energy_total": float(self._total_energy()),
            "energy_drift": float(self._total_energy() - self.total_energy_ref),
        }
        self.metrics.update(stats, events)
        return StepResult(stats=stats, events=events)

    def run(self, steps: int) -> list[StepResult]:
        out: list[StepResult] = []
        for _ in range(max(1, int(steps))):
            out.append(self.step())
        return out

    def encode_prompt_tokens(self, prompt: str) -> np.ndarray:
        text = str(prompt or "")
        if self.cfg.token_space == "gpt2":
            ids = encode_gpt2_tokens(text, self.cfg.gpt2_model_name)
        else:
            ids = encode_utf8_parity_tokens(text)
        if not ids:
            return np.zeros((0,), dtype=np.int32)
        arr = np.asarray(ids, dtype=np.int32)
        return np.clip(arr, 0, self.base_vocab_size - 1).astype(np.int32)

    def decode_tokens_text(self, token_ids: list[int] | np.ndarray, max_chars: int = 2000) -> str:
        ids = [int(x) for x in np.asarray(token_ids, dtype=np.int32).tolist()]
        if not ids:
            return ""
        if self.cfg.token_space == "gpt2":
            text = decode_gpt2_tokens(ids, self.cfg.gpt2_model_name)
        else:
            raw = bytes((int(t) & 0xFF) for t in ids if 0 <= int(t) < self.base_vocab_size)
            text = raw.decode("utf-8", errors="replace")
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def _infer_identity_tokens_for_frontier(self, identity_bytes: np.ndarray, frontier_tid: int) -> np.ndarray:
        b = np.asarray(identity_bytes, dtype=np.int32).reshape(-1)
        if b.size == 0:
            return np.zeros((0,), dtype=np.int32)
        frontier = int(np.clip(frontier_tid, 0, self.base_vocab_size - 1))
        if self.cfg.token_space == "byte_parity":
            frontier_symbol = frontier & 0xFF
            offsets = np.where(np.clip(b, 0, 255) == frontier_symbol)[0].astype(np.int32)
        else:
            offsets = np.where(np.clip(b, 0, self.base_vocab_size - 1) == frontier)[0].astype(np.int32)
        if offsets.size == 0:
            suffix = b[: max(1, min(b.size, self.cfg.proposal_lmax))]
            if self.cfg.token_space == "byte_parity":
                suffix = np.clip(suffix, 0, 255).astype(np.int32)
                p0 = (frontier // 256) & 1
                parity = (p0 + np.arange(suffix.size, dtype=np.int32)) & 1
                return (suffix + 256 * parity).astype(np.int32)
            return np.clip(suffix, 0, self.base_vocab_size - 1).astype(np.int32)

        best = np.zeros((0,), dtype=np.int32)
        best_len = -1
        for off in offsets.tolist():
            suffix = b[int(off) : int(off) + int(self.cfg.proposal_lmax)]
            if suffix.size <= 0:
                continue
            if self.cfg.token_space == "byte_parity":
                suffix = np.clip(suffix, 0, 255).astype(np.int32)
                p0 = (frontier // 256) & 1
                parity = (p0 + np.arange(suffix.size, dtype=np.int32)) & 1
                ids = (suffix + 256 * parity).astype(np.int32)
            else:
                ids = np.clip(suffix, 0, self.base_vocab_size - 1).astype(np.int32)
            if int(ids.size) > best_len:
                best = ids
                best_len = int(ids.size)
        return best

    def _infer_next_token(
        self,
        frontier_tid: int,
        right_tid: int | None = None,
        top_k: int = 16,
        temperature: float = 0.0,
        recent_tokens: list[int] | None = None,
    ) -> tuple[int, dict]:
        candidates: dict[int, float] = {}
        details: list[tuple[int, int, float]] = []
        if not self.super_tokens:
            fallback = int(np.clip(frontier_tid, 0, self.base_vocab_size - 1))
            return fallback, {"used": 0, "top": []}

        pool = sorted(
            self.super_tokens.values(),
            key=lambda t: float(t.energy),
            reverse=True,
        )[: max(1, int(top_k) * 8)]
        frontier = int(np.clip(frontier_tid, 0, self.base_vocab_size - 1))
        right = None if right_tid is None else int(np.clip(right_tid, 0, self.base_vocab_size - 1))
        for token in pool:
            ids = self._infer_identity_tokens_for_frontier(token.identity_bytes, frontier)
            if ids.size <= 0:
                continue
            if ids.size >= 2:
                nxt = int(ids[1])
            else:
                nxt = int(ids[0])
            energy_norm = float(np.clip(float(token.energy) / max(float(self.cfg.token_energy_cap), 1e-6), 0.0, 1.0))
            length_norm = float(np.clip(float(ids.size) / max(float(self.cfg.proposal_lmax), 1.0), 0.0, 1.0))
            conf = 0.30 + 0.45 * length_norm + 0.25 * energy_norm
            score = float(np.clip(conf * (0.25 + 0.75 * energy_norm), 0.0, 5.0))
            if right is not None and ids.size >= 3 and int(ids[2]) == right:
                score += 0.35
            elif right is not None and nxt == right:
                score += 0.10
            candidates[nxt] = candidates.get(nxt, 0.0) + score
            details.append((int(token.token_id), nxt, score))

        if not candidates:
            fallback = int(np.clip(frontier, 0, self.base_vocab_size - 1))
            return fallback, {"used": 0, "top": []}

        if recent_tokens:
            tail = recent_tokens[-32:]
            freq: dict[int, int] = {}
            for t in tail:
                tid = int(t)
                freq[tid] = freq.get(tid, 0) + 1
            for tid, count in freq.items():
                if tid in candidates:
                    candidates[tid] -= 0.18 * float(count)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        if float(temperature) <= 1e-8:
            choice = int(ranked[0][0])
        else:
            top = ranked[: max(1, int(top_k))]
            toks = np.asarray([int(t) for t, _ in top], dtype=np.int32)
            vals = np.asarray([float(s) for _, s in top], dtype=np.float64)
            vals = vals - float(np.max(vals))
            probs = np.exp(vals / max(float(temperature), 1e-6))
            probs = probs / max(float(np.sum(probs)), 1e-12)
            idx = int(self.rng.choice(len(toks), p=probs))
            choice = int(toks[idx])

        preview = [(int(t), float(s)) for t, s in ranked[: max(1, int(top_k))]]
        return choice, {"used": int(len(details)), "top": preview}

    def infer_generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        diffusion_passes: int = 2,
        window_size: int = 16,
        top_k: int = 16,
        temperature: float = 0.0,
    ) -> dict:
        prompt_ids = self.encode_prompt_tokens(prompt)
        if prompt_ids.size <= 0:
            prompt_ids = np.asarray([int(np.clip(self.world.peek(1)[0], 0, self.base_vocab_size - 1))], dtype=np.int32)

        seq: list[int] = [int(x) for x in prompt_ids.tolist()]
        max_new = max(1, int(max_new_tokens))
        passes = max(1, int(diffusion_passes))
        win = max(1, int(window_size))
        prompt_len = int(len(seq))

        debug_steps: list[str] = []

        # Initial autoregressive fill.
        for i in range(max_new):
            frontier = seq[-1]
            nxt, diag = self._infer_next_token(
                frontier_tid=frontier,
                right_tid=None,
                top_k=top_k,
                temperature=temperature,
                recent_tokens=seq,
            )
            seq.append(int(nxt))
            if i < 8:
                debug_steps.append(
                    f"gen[{i}] frontier={frontier} -> {nxt} cands={diag['used']}"
                )

        # Sliding-window diffusion refinement over generated segment only.
        final_len = len(seq)
        gen_start = prompt_len
        gen_end = final_len
        for pass_idx in range(1, passes):
            for start in range(gen_start, gen_end):
                end = min(start + win, gen_end)
                for pos in range(start, end):
                    left = seq[pos - 1] if pos > 0 else seq[0]
                    right = seq[pos + 1] if pos + 1 < len(seq) else None
                    nxt, _ = self._infer_next_token(
                        frontier_tid=left,
                        right_tid=right,
                        top_k=top_k,
                        temperature=temperature,
                        recent_tokens=seq[:pos],
                    )
                    seq[pos] = int(nxt)
            if pass_idx <= 2:
                debug_steps.append(f"diffusion_pass[{pass_idx}] done")

        out = seq[prompt_len:]
        return {
            "token_space": self.cfg.token_space,
            "prompt_tokens": int(prompt_len),
            "generated_tokens": int(len(out)),
            "passes": int(passes),
            "window": int(win),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "output_token_ids": [int(x) for x in out],
            "output_text": self.decode_tokens_text(out, max_chars=4000),
            "debug": debug_steps[:16],
        }
