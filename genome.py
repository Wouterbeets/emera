from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EmeraConfig


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _identity_symbol_vocab(cfg: EmeraConfig) -> int:
    if cfg.token_space == "byte_parity":
        return 256
    return max(int(cfg.base_tokens), 1)


_ROOT_IDENTITY_LEN = 2


def _coerce_root_pair(
    identity_bytes: np.ndarray | None,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vocab = max(int(vocab_size), 1)
    raw = np.asarray(identity_bytes, dtype=np.int32).reshape(-1) if identity_bytes is not None else np.zeros((0,), dtype=np.int32)
    if raw.size >= _ROOT_IDENTITY_LEN:
        out = raw[:_ROOT_IDENTITY_LEN].copy()
    elif raw.size == 1:
        out = np.asarray([int(raw[0]), int(rng.integers(0, vocab))], dtype=np.int32)
    else:
        out = rng.integers(0, vocab, size=(_ROOT_IDENTITY_LEN,), dtype=np.int32)
    return np.clip(out, 0, max(vocab - 1, 0)).astype(np.int32, copy=False)


def _compose_recursive_lineage_identity(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    max_len: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a = np.asarray(parent_a, dtype=np.int32).reshape(-1)
    b = np.asarray(parent_b, dtype=np.int32).reshape(-1)
    if a.size <= 0:
        a = _coerce_root_pair(None, vocab_size, rng)
    if b.size <= 0:
        b = _coerce_root_pair(None, vocab_size, rng)
    limit = max(2, int(max_len))
    if int(a.size + b.size) <= limit:
        out = np.concatenate([a, b], axis=0)
    else:
        keep_a = max(1, min(int(a.size), limit // 2))
        keep_b = max(1, min(int(b.size), limit - keep_a))
        rem = max(limit - keep_a - keep_b, 0)
        if rem > 0:
            room_a = max(int(a.size) - keep_a, 0)
            add_a = min(rem, room_a)
            keep_a += add_a
            rem -= add_a
        if rem > 0:
            room_b = max(int(b.size) - keep_b, 0)
            add_b = min(rem, room_b)
            keep_b += add_b
        out = np.concatenate([a[-keep_a:], b[:keep_b]], axis=0)
    return np.clip(out[:limit], 0, max(int(vocab_size) - 1, 0)).astype(np.int32, copy=False)


@dataclass
class Proposal:
    token_id: int
    tokens: np.ndarray
    confidence: float
    quality: float
    expected_return: float
    bet: float


@dataclass
class SuperToken:
    token_id: int
    parent_a: int
    parent_b: int
    energy: float
    inactivity_steps: int
    state_vec: np.ndarray
    signature: np.ndarray
    proposal_drift: np.ndarray
    ifs: np.ndarray  # [num_ifs, 2, 3]
    phase: float
    omega: float
    activation_threshold: float
    emission_amplitude: float
    emission_decay: float
    silence_growth_rate: float
    resonance_width: float
    phase_coupling: float
    velocity_coupling: float
    proposal_length_bias: float
    identity_bytes: np.ndarray
    birth_step: int
    low_energy_steps: int
    max_paid_bet: float
    max_silent_correct: int

    def chaos_step(self, rng: np.random.Generator, resonance_latent: np.ndarray) -> np.ndarray:
        resonance_latent = np.nan_to_num(resonance_latent, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        idx = int(rng.integers(0, self.ifs.shape[0]))
        a = self.ifs[idx, :, :2]
        b = self.ifs[idx, :, 2]

        prev = self.state_vec[:2].copy()
        nxt = np.tanh(a @ prev + b + 0.12 * resonance_latent[:2]).astype(np.float32)
        vel2 = (nxt - prev).astype(np.float32)

        self.state_vec[:2] = nxt
        self.state_vec = np.tanh(
            (1.0 - 0.3 * np.clip(self.emission_decay, 0.0, 1.0)) * self.state_vec
            + 0.22 * resonance_latent
        ).astype(np.float32)
        self.state_vec = np.nan_to_num(self.state_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.phase = float(np.arctan2(self.state_vec[1], self.state_vec[0]))
        return vel2

    def emission(self, cfg: EmeraConfig, vel2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        emit = np.tanh(
            self.emission_amplitude * self.state_vec[: cfg.gap_dim]
            + cfg.beacon_strength * self.signature
        ).astype(np.float32)
        vel = np.zeros((cfg.gap_dim,), dtype=np.float32)
        vel[:2] = vel2.astype(np.float32)
        vel = np.tanh(vel * (0.5 + self.velocity_coupling)).astype(np.float32)
        eng = max(self.emission_amplitude * max(self.energy, 0.0) * 0.05, 1e-6)
        return emit, vel, float(eng)

    def proposal_length(self, cfg: EmeraConfig, resonance_strength: float) -> int:
        if not np.isfinite(resonance_strength):
            resonance_strength = 0.0
        x = self.proposal_length_bias + 2.2 * float(resonance_strength)
        if not np.isfinite(x):
            x = 0.0
        frac = _sigmoid(x)
        l = 1 + int(np.floor(frac * float(cfg.proposal_lmax - 1)))
        return int(np.clip(l, 1, cfg.proposal_lmax))

    def decode_tokens(
        self,
        base_latent: np.ndarray,
        cfg: EmeraConfig,
        resonance_strength: float,
        candidate_token_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        length = self.proposal_length(cfg, resonance_strength)
        ids = np.zeros((length,), dtype=np.int32)
        confs = np.zeros((length,), dtype=np.float32)
        if candidate_token_ids is None:
            candidate_token_ids = np.arange(base_latent.shape[0], dtype=np.int32)
        q = self.state_vec.copy()
        drift = self.proposal_drift
        for k in range(length):
            query = _normalize(q + float(k) * drift)
            scores = base_latent @ query
            top2 = np.argpartition(scores, -2)[-2:]
            if scores[top2[0]] >= scores[top2[1]]:
                i1, i2 = int(top2[0]), int(top2[1])
            else:
                i1, i2 = int(top2[1]), int(top2[0])
            margin = float(scores[i1] - scores[i2])
            conf = _sigmoid(cfg.confidence_scale * margin)
            ids[k] = int(candidate_token_ids[i1])
            confs[k] = float(conf)
        confidence = float(np.mean(confs))
        quality = float(confidence * (0.5 + 0.5 * np.clip(resonance_strength, 0.0, 1.0)))
        return ids, confidence, quality


def _template_ifs(cfg: EmeraConfig) -> np.ndarray:
    out = np.zeros((cfg.num_ifs, 2, 3), dtype=np.float32)
    for k in range(cfg.num_ifs):
        angle = (k / max(cfg.num_ifs, 1)) * 2.0 * np.pi
        scale = 0.65
        c = np.cos(angle) * scale
        s = np.sin(angle) * scale
        out[k, :, :2] = np.array([[c, -s], [s, c]], dtype=np.float32)
        out[k, :, 2] = np.array([0.05 * np.cos(angle), 0.05 * np.sin(angle)], dtype=np.float32)
    return out


def _sample_state_vec(cfg: EmeraConfig, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, size=(cfg.d_latent,)).astype(np.float32)
    return _normalize(v)


def _sample_drift(cfg: EmeraConfig, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, cfg.proposal_drift_scale, size=(cfg.d_latent,)).astype(np.float32)
    return _normalize(v)


def make_initial_super_token(
    cfg: EmeraConfig,
    rng: np.random.Generator,
    token_id: int,
    birth_step: int = 0,
    identity_bytes: np.ndarray | None = None,
) -> SuperToken:
    state = _sample_state_vec(cfg, rng)
    sig = _normalize(state[: cfg.gap_dim]).astype(np.float32)
    ifs = (_template_ifs(cfg) + rng.normal(0.0, cfg.ifs_mutation_scale, size=(cfg.num_ifs, 2, 3))).astype(np.float32)
    omega = float(cfg.omega_base + cfg.omega_jitter * rng.normal())
    ident = _coerce_root_pair(identity_bytes, _identity_symbol_vocab(cfg), rng)
    return SuperToken(
        token_id=int(token_id),
        parent_a=-1,
        parent_b=-1,
        energy=float(cfg.initial_super_energy),
        inactivity_steps=0,
        state_vec=state.astype(np.float32),
        signature=sig.astype(np.float32),
        proposal_drift=_sample_drift(cfg, rng).astype(np.float32),
        ifs=ifs,
        phase=float(np.arctan2(state[1], state[0])),
        omega=omega,
        activation_threshold=float(cfg.activation_threshold_init),
        emission_amplitude=float(cfg.emission_amplitude_init),
        emission_decay=float(cfg.emission_decay_init),
        silence_growth_rate=float(cfg.silence_growth_init),
        resonance_width=float(cfg.resonance_width_init),
        phase_coupling=float(cfg.phase_coupling_init),
        velocity_coupling=float(cfg.velocity_coupling_init),
        proposal_length_bias=float(cfg.proposal_length_bias_init),
        identity_bytes=ident,
        birth_step=int(birth_step),
        low_energy_steps=0,
        max_paid_bet=0.0,
        max_silent_correct=0,
    )


def create_initial_population(
    cfg: EmeraConfig,
    rng: np.random.Generator,
    start_token_id: int,
    birth_step: int = 0,
    identity_bank: list[np.ndarray] | None = None,
) -> dict[int, SuperToken]:
    out: dict[int, SuperToken] = {}
    for i in range(cfg.initial_super_tokens):
        tid = start_token_id + i
        ident = None
        if identity_bank is not None and i < len(identity_bank):
            ident = identity_bank[i]
        out[tid] = make_initial_super_token(cfg, rng, tid, birth_step=birth_step, identity_bytes=ident)
    return out


def _mutate_scalar(v: float, scale: float, lo: float, hi: float, rng: np.random.Generator) -> float:
    m = float(v * np.exp(rng.normal(0.0, scale)))
    return float(np.clip(m, lo, hi))


def _sample_pareto_factor(
    rng: np.random.Generator,
    alpha: float,
    scale: float,
    clip_max: float,
) -> float:
    a = max(float(alpha), 1.01)
    raw = scale * (rng.pareto(a) + 1.0)
    raw = float(np.clip(raw, 1e-6, clip_max))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(np.exp(sign * (raw - 1.0)))


def _mutate_identity_bytes(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    max_len: int,
    vocab_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a = np.asarray(parent_a, dtype=np.int32).reshape(-1)
    b = np.asarray(parent_b, dtype=np.int32).reshape(-1)
    if a.size == 0:
        a = np.asarray([32], dtype=np.int32)
    if b.size == 0:
        b = np.asarray([32], dtype=np.int32)

    modes = int(rng.integers(0, 5))
    if modes == 0:
        cand = a.copy()
    elif modes == 1:
        cand = b.copy()
    elif modes == 2:
        na = max(1, a.size // 2)
        nb = max(1, b.size - b.size // 2)
        cand = np.concatenate([a[:na], b[-nb:]], axis=0)
    elif modes == 3:
        nb = max(1, b.size // 2)
        na = max(1, a.size - a.size // 2)
        cand = np.concatenate([b[:nb], a[-na:]], axis=0)
    else:
        cand = np.concatenate([a, b], axis=0)

    limit = max(1, int(max_len))
    if cand.size > limit:
        start = int(rng.integers(0, cand.size - limit + 1))
        cand = cand[start : start + limit]
    if cand.size == 0:
        cand = np.asarray([32], dtype=np.int32)

    if rng.random() < 0.15:
        idx = int(rng.integers(0, cand.size))
        cand[idx] = int(rng.integers(0, max(int(vocab_size), 1)))
    if cand.size < limit and rng.random() < 0.15:
        cand = np.concatenate(
            [cand, np.asarray([int(rng.integers(0, max(int(vocab_size), 1)))], dtype=np.int32)],
            axis=0,
        )

    return np.clip(cand, 0, max(int(vocab_size) - 1, 0)).astype(np.int32)


def mint_child_from_parents(
    cfg: EmeraConfig,
    rng: np.random.Generator,
    new_id: int,
    parent_a: SuperToken,
    parent_b: SuperToken,
    spawn_energy: float,
    pareto_alpha: float,
    identity_override: np.ndarray | None = None,
    ifs_override: np.ndarray | None = None,
    birth_step: int = 0,
) -> SuperToken:
    state = _normalize(
        0.5 * (parent_a.state_vec + parent_b.state_vec)
        + rng.normal(0.0, cfg.mutation_scale, size=parent_a.state_vec.shape).astype(np.float32)
    ).astype(np.float32)
    sig = _normalize(state[: cfg.gap_dim]).astype(np.float32)

    parent_ifs_seed = 0.5 * (parent_a.ifs + parent_b.ifs)
    if ifs_override is None:
        ifs_seed = parent_ifs_seed.astype(np.float32)
    else:
        raw = np.asarray(ifs_override, dtype=np.float32).reshape(-1)
        need = int(cfg.num_ifs) * 6
        packed = np.zeros((need,), dtype=np.float32)
        n = min(int(raw.size), need)
        if n > 0:
            packed[:n] = raw[:n]
        ov = packed.reshape(int(cfg.num_ifs), 2, 3)
        ifs_seed = (0.75 * ov + 0.25 * parent_ifs_seed).astype(np.float32)
    child_ifs = (
        ifs_seed
        + rng.normal(0.0, cfg.ifs_mutation_scale, size=parent_a.ifs.shape).astype(np.float32)
    ).astype(np.float32)
    drift = _normalize(
        0.5 * (parent_a.proposal_drift + parent_b.proposal_drift)
        + rng.normal(0.0, cfg.proposal_drift_scale, size=parent_a.proposal_drift.shape).astype(np.float32)
    ).astype(np.float32)
    f = [
        _sample_pareto_factor(
            rng=rng,
            alpha=pareto_alpha,
            scale=cfg.pareto_mutation_scale,
            clip_max=cfg.pareto_clip,
        )
        for _ in range(8)
    ]

    vocab = _identity_symbol_vocab(cfg)
    ident = _compose_recursive_lineage_identity(
        parent_a=parent_a.identity_bytes,
        parent_b=parent_b.identity_bytes,
        max_len=cfg.proposal_lmax,
        vocab_size=vocab,
        rng=rng,
    )
    if identity_override is not None:
        ov = np.asarray(identity_override, dtype=np.int32).reshape(-1)
        if ov.size > 0 and ident.size > 0:
            ov = np.clip(ov, 0, max(vocab - 1, 0)).astype(np.int32, copy=False)
            edits = max(1, min(int(ident.size // 4), int(ov.size)))
            idx = np.asarray(rng.choice(int(ident.size), size=edits, replace=False), dtype=np.int32)
            for j, pos in enumerate(idx.tolist()):
                ident[pos] = int(ov[j % int(ov.size)])
            ident = np.clip(ident, 0, max(vocab - 1, 0)).astype(np.int32, copy=False)

    return SuperToken(
        token_id=int(new_id),
        parent_a=int(parent_a.token_id),
        parent_b=int(parent_b.token_id),
        energy=float(spawn_energy),
        inactivity_steps=0,
        state_vec=state,
        signature=sig,
        proposal_drift=drift,
        ifs=child_ifs,
        phase=float(np.arctan2(state[1], state[0])),
        omega=float(np.clip(0.5 * (parent_a.omega + parent_b.omega), 0.0, np.pi / 2.0)),
        activation_threshold=_mutate_scalar(
            0.5 * (parent_a.activation_threshold + parent_b.activation_threshold) * f[0],
            cfg.mutation_scale,
            0.01,
            0.99,
            rng,
        ),
        emission_amplitude=_mutate_scalar(
            0.5 * (parent_a.emission_amplitude + parent_b.emission_amplitude) * f[1],
            cfg.mutation_scale,
            0.05,
            3.0,
            rng,
        ),
        emission_decay=_mutate_scalar(
            0.5 * (parent_a.emission_decay + parent_b.emission_decay) * f[2],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        silence_growth_rate=_mutate_scalar(
            0.5 * (parent_a.silence_growth_rate + parent_b.silence_growth_rate) * f[3],
            cfg.mutation_scale,
            0.01,
            8.0,
            rng,
        ),
        resonance_width=_mutate_scalar(
            0.5 * (parent_a.resonance_width + parent_b.resonance_width) * f[4],
            cfg.mutation_scale,
            0.05,
            2.0,
            rng,
        ),
        phase_coupling=_mutate_scalar(
            0.5 * (parent_a.phase_coupling + parent_b.phase_coupling) * f[5],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        velocity_coupling=_mutate_scalar(
            0.5 * (parent_a.velocity_coupling + parent_b.velocity_coupling) * f[6],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        proposal_length_bias=_mutate_scalar(
            0.5 * (parent_a.proposal_length_bias + parent_b.proposal_length_bias) * f[7],
            cfg.mutation_scale,
            -3.0,
            3.0,
            rng,
        ),
        identity_bytes=ident,
        birth_step=int(birth_step),
        low_energy_steps=0,
        max_paid_bet=0.0,
        max_silent_correct=0,
    )


def mint_child_from_parent(
    cfg: EmeraConfig,
    rng: np.random.Generator,
    new_id: int,
    parent: SuperToken,
    spawn_energy: float,
    pareto_alpha: float,
    identity_override: np.ndarray | None = None,
    ifs_override: np.ndarray | None = None,
    birth_step: int = 0,
) -> SuperToken:
    state = _normalize(
        parent.state_vec
        + rng.normal(0.0, cfg.mutation_scale, size=parent.state_vec.shape).astype(np.float32)
    ).astype(np.float32)
    sig = _normalize(state[: cfg.gap_dim]).astype(np.float32)

    parent_ifs_seed = parent.ifs
    if ifs_override is None:
        ifs_seed = parent_ifs_seed.astype(np.float32)
    else:
        raw = np.asarray(ifs_override, dtype=np.float32).reshape(-1)
        need = int(cfg.num_ifs) * 6
        packed = np.zeros((need,), dtype=np.float32)
        n = min(int(raw.size), need)
        if n > 0:
            packed[:n] = raw[:n]
        ov = packed.reshape(int(cfg.num_ifs), 2, 3)
        ifs_seed = (0.80 * ov + 0.20 * parent_ifs_seed).astype(np.float32)
    child_ifs = (
        ifs_seed
        + rng.normal(0.0, cfg.ifs_mutation_scale, size=parent.ifs.shape).astype(np.float32)
    ).astype(np.float32)
    drift = _normalize(
        parent.proposal_drift
        + rng.normal(0.0, cfg.proposal_drift_scale, size=parent.proposal_drift.shape).astype(np.float32)
    ).astype(np.float32)
    f = [
        _sample_pareto_factor(
            rng=rng,
            alpha=pareto_alpha,
            scale=cfg.pareto_mutation_scale,
            clip_max=cfg.pareto_clip,
        )
        for _ in range(8)
    ]

    if identity_override is None:
        ident = _mutate_identity_bytes(
            parent_a=parent.identity_bytes,
            parent_b=parent.identity_bytes,
            max_len=cfg.proposal_lmax,
            vocab_size=_identity_symbol_vocab(cfg),
            rng=rng,
        )
    else:
        ident = np.asarray(identity_override, dtype=np.int32).reshape(-1)
        if ident.size == 0:
            ident = _mutate_identity_bytes(
                parent_a=parent.identity_bytes,
                parent_b=parent.identity_bytes,
                max_len=cfg.proposal_lmax,
                vocab_size=_identity_symbol_vocab(cfg),
                rng=rng,
            )
        else:
            vmax = _identity_symbol_vocab(cfg) - 1
            ident = np.clip(
                ident[: max(int(cfg.proposal_lmax), 1)],
                0,
                max(vmax, 0),
            ).astype(np.int32)

    return SuperToken(
        token_id=int(new_id),
        parent_a=int(parent.token_id),
        parent_b=-1,
        energy=float(spawn_energy),
        inactivity_steps=0,
        state_vec=state,
        signature=sig,
        proposal_drift=drift,
        ifs=child_ifs,
        phase=float(np.arctan2(state[1], state[0])),
        omega=_mutate_scalar(
            parent.omega,
            cfg.mutation_scale,
            0.0,
            np.pi / 2.0,
            rng,
        ),
        activation_threshold=_mutate_scalar(
            parent.activation_threshold * f[0],
            cfg.mutation_scale,
            0.01,
            0.99,
            rng,
        ),
        emission_amplitude=_mutate_scalar(
            parent.emission_amplitude * f[1],
            cfg.mutation_scale,
            0.05,
            3.0,
            rng,
        ),
        emission_decay=_mutate_scalar(
            parent.emission_decay * f[2],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        silence_growth_rate=_mutate_scalar(
            parent.silence_growth_rate * f[3],
            cfg.mutation_scale,
            0.01,
            8.0,
            rng,
        ),
        resonance_width=_mutate_scalar(
            parent.resonance_width * f[4],
            cfg.mutation_scale,
            0.05,
            2.0,
            rng,
        ),
        phase_coupling=_mutate_scalar(
            parent.phase_coupling * f[5],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        velocity_coupling=_mutate_scalar(
            parent.velocity_coupling * f[6],
            cfg.mutation_scale,
            0.0,
            1.0,
            rng,
        ),
        proposal_length_bias=_mutate_scalar(
            parent.proposal_length_bias * f[7],
            cfg.mutation_scale,
            -3.0,
            3.0,
            rng,
        ),
        identity_bytes=ident,
        birth_step=int(birth_step),
        low_energy_steps=0,
        max_paid_bet=0.0,
        max_silent_correct=0,
    )
