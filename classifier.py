"""Emera as a classifier: a typed-decision substrate built from the same parts.

The book-world engine plays "guess the next byte of an infinite stream". This
module keeps the substrate — chaos-game organisms, the gap field as a shared
medium, an energy ledger that kills what does not pay for itself, symbiogenesis
— and swaps the game for "guess the label of the example in front of you".

What changes, and why it matters:

* The world becomes a labelled example stream. Ground truth is one label per
  step instead of a byte at a cursor, so credit assignment is dense and exact.
* The readout is geometric again. `engine.py` decodes proposals by literal
  byte alignment (`_decode_from_identity`); its geometric decoder is dead code.
  Here an organism votes by projecting its state onto a label codebook, so the
  chaos dynamics carry real predictive load.
* Payout is parimutuel instead of a jackpot table. Losers' stakes fund winners'
  payouts, which conserves energy exactly and prices each label by how crowded
  it is: being right when the crowd is wrong is what pays.
* Inactivity is abstention. An organism that does not recognise the example
  pays nothing and collects silence credit, so narrow specialists are viable.

An organism is therefore a rule: *a pattern it recognises, a label it leans
toward, and how much it is willing to stake on that*. The whole population can
be printed and read, which is the main thing this buys over a dense encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from config import EmeraConfig
from gap_field import GapField
from genome import (
    SuperToken,
    _mutate_scalar,
    _normalize,
    _sample_drift,
    _sample_pareto_factor,
    _sample_state_vec,
    _template_ifs,
)
from identity import (
    create_base_identity,
    hadamard_matrix,
    random_orthonormal_projection,
)
from tasks import Example, Task


@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier-only knobs. Substrate geometry comes from `EmeraConfig`."""

    seed: int = 42

    # Population
    initial_population: int = 512
    max_population: int = 4096
    initial_energy: float = 2.0
    min_viable_energy: float = 1e-3
    newborn_grace_steps: int = 50

    # Economy (all transfers are organism <-> organism or organism <-> reservoir)
    reservoir_init: float = 9000.0
    emit_cost: float = 0.004
    metabolic_tax_rate: float = 0.0015
    stake_max_energy_frac: float = 0.10
    stake_min: float = 0.002
    rake_frac: float = 0.06  # skimmed from each pot to fund births and silence
    energy_cap: float = 40.0
    # Metabolism and silence are slow processes, so they run on a slower clock:
    # every `economy_interval` steps the whole population is charged and fed at
    # once. Transfers stay exact; only the sweep over dormant organisms is
    # amortised, which is what keeps a large population affordable.
    economy_interval: int = 16

    # Silence credit (accelerating, as in the book-world ledger).
    # Subsistence for the dormant poor, not a savings account: above
    # `silence_energy_ceiling` an organism is already fed and collects nothing,
    # which is what stops "abstain forever" from being a winning strategy.
    silence_log_coeff: float = 0.010
    silence_exp_coeff: float = 0.0016
    silence_exp_rate: float = 0.020
    silence_cap: float = 0.05
    silence_energy_ceiling: float = 1.2

    # Waking: how much of the decision to activate comes from the literal
    # pattern match vs. from resonance with what other organisms just emitted.
    wake_feature_weight: float = 0.80
    wake_resonance_weight: float = 0.20

    # Voting geometry
    activation_threshold_init: float = 0.30
    context_gain_init: float = 0.35
    vote_margin_scale: float = 6.0
    readout_temperature: float = 0.60
    # Wealth alone is a noisy fitness signal: staking a fixed fraction of
    # energy on a near-fair market is a multiplicative process, so survivors
    # look rich whether or not they were right. Weighting a vote by realised
    # return on stake - still nothing but the organism's own energy history -
    # separates the rules that earned their keep from the ones that got lucky.
    roi_prior: float = 0.5
    roi_max: float = 4.0

    # Reproduction
    spawn_energy: float = 1.2
    self_copy_cost: float = 0.45
    self_copy_min_energy: float = 4.0
    self_copy_max_per_step: int = 2
    discovery_spawn_max_per_step: int = 2
    discovery_spawn_prob: float = 0.85
    discovery_reuse_prob: float = 0.50
    mutation_scale: float = 0.10
    vote_mutation_scale: float = 0.10
    pareto_alpha: float = 1.9
    pareto_scale: float = 0.22
    pareto_clip: float = 12.0

    # Pattern genome. `pattern_index_len` is the prefix length used to find the
    # organisms a given example could possibly wake; patterns are never shorter
    # than it, so the index never hides a match that mattered.
    pattern_min_len: int = 4
    pattern_max_len: int = 8
    pattern_index_len: int = 4

    def validate(self) -> None:
        if self.initial_population < 1:
            raise ValueError("initial_population must be >= 1.")
        if self.max_population < self.initial_population:
            raise ValueError("max_population must be >= initial_population.")
        if not (0.0 < self.stake_max_energy_frac <= 1.0):
            raise ValueError("stake_max_energy_frac must be in (0, 1].")
        if not (0.0 <= self.rake_frac < 1.0):
            raise ValueError("rake_frac must be in [0, 1).")
        if self.pattern_min_len < 1 or self.pattern_max_len < self.pattern_min_len:
            raise ValueError("invalid pattern length bounds.")
        if self.pattern_index_len > self.pattern_min_len:
            raise ValueError("pattern_index_len must be <= pattern_min_len.")
        if self.readout_temperature <= 0.0:
            raise ValueError("readout_temperature must be > 0.")


@dataclass
class Voter(SuperToken):
    """A super-token that bets on labels instead of proposing byte segments."""

    vote_bias: np.ndarray = field(default_factory=lambda: np.zeros((1,), np.float32))
    context_gain: float = 0.35
    stake_scale: float = 1.0
    wins: int = 0
    losses: int = 0
    staked_total: float = 0.0
    won_total: float = 0.0
    votes: int = 0
    pattern: bytes = b""
    last_active_step: int = 0

    def query(self) -> np.ndarray:
        return _normalize(self.vote_bias + self.context_gain * self.state_vec)

    def refresh_pattern(self) -> bytes:
        self.pattern = _pattern_bytes(self.identity_bytes)
        return self.pattern


@dataclass
class Vote:
    token_id: int
    label: int
    confidence: float
    stake: float


@dataclass
class StepReport:
    step: int
    correct: bool
    predicted: int
    truth: int
    voters: int
    abstained: int
    pot: float
    winners: int
    births: int
    deaths: int
    population: int
    reservoir: float
    total_energy: float


def _pattern_bytes(ident: np.ndarray) -> bytes:
    arr = np.asarray(ident, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return b""
    return bytes(int(v) & 0xFF for v in arr.tolist())


def _clip_pattern(
    pattern: np.ndarray, fallback: np.ndarray, cfg: ClassifierConfig
) -> np.ndarray:
    """Keep a pattern inside the length bounds the prefix index relies on."""
    arr = np.asarray(pattern, dtype=np.int32).reshape(-1) & 0xFF
    if arr.size > cfg.pattern_max_len:
        arr = arr[: cfg.pattern_max_len]
    if arr.size < cfg.pattern_min_len:
        pad = np.asarray(fallback, dtype=np.int32).reshape(-1) & 0xFF
        arr = np.concatenate([arr, pad])[: cfg.pattern_max_len]
    if arr.size < cfg.pattern_min_len:
        arr = np.concatenate(
            [arr, np.full((cfg.pattern_min_len - arr.size,), 32, dtype=np.int32)]
        )
    return arr.astype(np.int32)


def _mutate_pattern(
    parent: np.ndarray, rng: np.random.Generator, cfg: ClassifierConfig
) -> np.ndarray:
    """Mutate a literal text pattern.

    The book-world's `_mutate_identity_bytes` composes parents by concatenation,
    which is right when a genome is a token sequence to be proposed but wrong
    here: a doubled pattern like `JudaJuda` never occurs in real text, so it can
    only ever match its own prefix. The moves that mean something for a detector
    are *generalise* (trim) and *perturb*; specialisation arrives instead from
    discovery spawns, which cut fresh patterns out of real examples.
    """
    arr = np.asarray(parent, dtype=np.int32).reshape(-1) & 0xFF
    if arr.size == 0:
        return _clip_pattern(arr, arr, cfg)
    roll = rng.random()
    if roll < 0.40 and arr.size > cfg.pattern_min_len:
        # Trim an end: a shorter pattern fires on more examples.
        if rng.random() < 0.5:
            arr = arr[1:]
        else:
            arr = arr[:-1]
    elif roll < 0.70:
        idx = int(rng.integers(0, arr.size))
        arr = arr.copy()
        arr[idx] = int(rng.integers(32, 127))
    elif roll < 0.85 and arr.size < cfg.pattern_max_len:
        idx = int(rng.integers(0, arr.size + 1))
        arr = np.insert(arr, idx, int(rng.integers(32, 127)))
    # The remaining 15% copy the parent pattern unchanged; the child still
    # differs in its vote lean and traits.
    return _clip_pattern(arr, parent, cfg)


def _sample_pattern(text: str, rng: np.random.Generator, cfg: ClassifierConfig) -> np.ndarray:
    """Draw a candidate pattern from an example, preferring word-shaped spans."""
    raw = text.encode("utf-8", errors="ignore")
    lo, hi = cfg.pattern_min_len, cfg.pattern_max_len
    if len(raw) < lo:
        return rng.integers(97, 123, size=(lo,), dtype=np.int32)
    words = [w for w in raw.split() if len(w) >= lo]
    if words and rng.random() < 0.75:
        word = words[int(rng.integers(0, len(words)))]
        length = int(rng.integers(lo, min(hi, len(word)) + 1))
        span = word[:length]
    else:
        length = int(rng.integers(lo, min(hi, len(raw)) + 1))
        start = int(rng.integers(0, max(len(raw) - length, 0) + 1))
        span = raw[start : start + length]
    return np.frombuffer(span, dtype=np.uint8).astype(np.int32)


class EmeraClassifier:
    """Population of chaos-game organisms betting on typed decisions."""

    def __init__(self, task: Task, cfg: EmeraConfig, ccfg: ClassifierConfig):
        cfg.validate()
        ccfg.validate()
        self.cfg = cfg
        self.ccfg = ccfg
        self.task = task
        self.num_labels = int(task.num_labels)
        self.rng = np.random.default_rng(ccfg.seed)

        self.base_identity = create_base_identity(cfg, self.rng)
        self.base_latent = self.base_identity.latent.astype(np.float32)
        self.label_latent = self._build_label_codebook()

        self.gap = GapField(cfg)
        self.reservoir = float(ccfg.reservoir_init)
        self.step_idx = 0
        self.next_token_id = int(self.base_latent.shape[0])
        self.population: dict[int, Voter] = {}
        # prefix -> organisms that could be woken by an example containing it
        self.prefix_index: dict[bytes, set[int]] = {}
        self._seed_population()
        self.total_energy_ref = self._total_energy()

        # Rolling diagnostics.
        self.recent_correct: list[int] = []
        self.births = 0
        self.deaths = 0

    # ---------------------------------------------------------------- setup

    def _build_label_codebook(self) -> np.ndarray:
        """Orthogonal-at-init latent per label, same scheme as base tokens."""
        # Hadamard rows must outnumber the latent dimensions for the random
        # orthonormal projection to keep full rank.
        size = 1
        while size < max(self.num_labels, self.cfg.d_latent, 2):
            size *= 2
        h = hadamard_matrix(size) / np.sqrt(float(size))
        proj = random_orthonormal_projection(self.rng, size, self.cfg.d_latent)
        z = h @ proj
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        return z[: self.num_labels].astype(np.float32)

    def _make_voter(
        self,
        pattern: np.ndarray,
        label_lean: int | None,
        energy: float,
        parent_a: int = -1,
        parent_b: int = -1,
    ) -> Voter:
        cfg = self.cfg
        state = _sample_state_vec(cfg, self.rng)
        if label_lean is None:
            bias = _normalize(self.rng.normal(0.0, 1.0, size=(cfg.d_latent,)).astype(np.float32))
        else:
            jitter = self.rng.normal(0.0, 0.25, size=(cfg.d_latent,)).astype(np.float32)
            bias = _normalize(self.label_latent[int(label_lean)] + jitter)
        ifs = (
            _template_ifs(cfg)
            + self.rng.normal(0.0, cfg.ifs_mutation_scale, size=(cfg.num_ifs, 2, 3))
        ).astype(np.float32)
        tid = self.next_token_id
        self.next_token_id += 1
        voter = Voter(
            token_id=int(tid),
            parent_a=int(parent_a),
            parent_b=int(parent_b),
            energy=float(energy),
            inactivity_steps=0,
            state_vec=state.astype(np.float32),
            signature=_normalize(state[: cfg.gap_dim]).astype(np.float32),
            proposal_drift=_sample_drift(cfg, self.rng).astype(np.float32),
            ifs=ifs,
            phase=float(np.arctan2(state[1], state[0])),
            omega=float(cfg.omega_base + cfg.omega_jitter * self.rng.normal()),
            activation_threshold=float(self.ccfg.activation_threshold_init),
            emission_amplitude=float(cfg.emission_amplitude_init),
            emission_decay=float(cfg.emission_decay_init),
            silence_growth_rate=float(cfg.silence_growth_init),
            resonance_width=float(cfg.resonance_width_init),
            phase_coupling=float(cfg.phase_coupling_init),
            velocity_coupling=float(cfg.velocity_coupling_init),
            proposal_length_bias=float(cfg.proposal_length_bias_init),
            identity_bytes=np.asarray(pattern, dtype=np.int32).reshape(-1),
            birth_step=int(self.step_idx),
            last_active_step=int(self.step_idx),
            low_energy_steps=0,
            max_paid_bet=0.0,
            max_silent_correct=0,
            vote_bias=bias.astype(np.float32),
            context_gain=float(self.ccfg.context_gain_init),
            stake_scale=1.0,
        )
        voter.identity_bytes = _clip_pattern(
            voter.identity_bytes, voter.identity_bytes, self.ccfg
        )
        voter.refresh_pattern()
        return voter

    def _seed_population(self) -> None:
        """Seed from the training pool: every organism starts life recognising
        something real, with a random label lean it has to earn or lose."""
        pool = self.task.train
        for _ in range(self.ccfg.initial_population):
            ex = pool[int(self.rng.integers(0, len(pool)))]
            pattern = _sample_pattern(ex.text, self.rng, self.ccfg)
            energy = self._reservoir_take(self.ccfg.initial_energy)
            voter = self._make_voter(pattern, label_lean=None, energy=energy)
            self._admit(voter)

    # ---------------------------------------------------------------- index

    def _prefix_key(self, voter: Voter) -> bytes:
        return voter.pattern[: self.ccfg.pattern_index_len]

    def _admit(self, voter: Voter) -> None:
        self.population[voter.token_id] = voter
        self.prefix_index.setdefault(self._prefix_key(voter), set()).add(
            voter.token_id
        )

    def _expel(self, voter: Voter) -> None:
        self.population.pop(voter.token_id, None)
        key = self._prefix_key(voter)
        bucket = self.prefix_index.get(key)
        if bucket is not None:
            bucket.discard(voter.token_id)
            if not bucket:
                self.prefix_index.pop(key, None)

    def _candidates(self, raw: bytes) -> dict[int, float]:
        """Organisms whose pattern prefix occurs in this example, with the
        graded match each one gets.

        An organism whose first `pattern_index_len` bytes are absent can match
        at most a shorter prefix than the index length, which cannot clear any
        usable activation threshold, so it is treated as asleep without being
        scored. This is what keeps a step proportional to the organisms the
        example actually concerns rather than to the whole population.
        """
        n = self.ccfg.pattern_index_len
        if len(raw) < n or not self.prefix_index:
            return {}
        seen: set[bytes] = {raw[i : i + n] for i in range(len(raw) - n + 1)}
        out: dict[int, float] = {}
        for key in seen & self.prefix_index.keys():
            for tid in self.prefix_index[key]:
                voter = self.population.get(tid)
                if voter is None:
                    continue
                out[tid] = self._feature_match(voter, raw)
        return out

    # -------------------------------------------------------------- economy

    def _total_energy(self) -> float:
        return float(self.reservoir + sum(v.energy for v in self.population.values()))

    def _reservoir_take(self, amount: float) -> float:
        take = float(min(max(amount, 0.0), self.reservoir))
        self.reservoir -= take
        return take

    def _reservoir_add(self, amount: float) -> None:
        self.reservoir += float(max(amount, 0.0))

    def _drain(self, voter: Voter, amount: float) -> float:
        paid = float(min(max(amount, 0.0), max(voter.energy, 0.0)))
        voter.energy -= paid
        return paid

    def _credit(self, voter: Voter, amount: float) -> float:
        room = max(self.ccfg.energy_cap - voter.energy, 0.0)
        given = float(min(max(amount, 0.0), room))
        voter.energy += given
        if given < amount:
            self._reservoir_add(float(amount) - given)
        return given

    def _silence_credit(self, voter: Voter) -> float:
        c = self.ccfg
        s = max(int(voter.inactivity_steps), 0)
        g = max(float(voter.silence_growth_rate), 1e-6)
        log_term = c.silence_log_coeff * g * float(np.log1p(s))
        expo = float(np.clip(c.silence_exp_rate * g * s, 0.0, 40.0))
        exp_term = c.silence_exp_coeff * (float(np.exp(expo)) - 1.0)
        return float(np.clip(log_term + exp_term, 0.0, c.silence_cap))

    # ------------------------------------------------------------ perception

    def _feature_match(self, voter: Voter, raw: bytes) -> float:
        """Longest prefix of the organism's pattern present in the example.

        Mirrors `_identity_prefix_fraction` in the book-world engine: graded
        recognition, not a binary hit, so partial matches still wake weakly.
        """
        pat = voter.pattern or voter.refresh_pattern()
        n = len(pat)
        if n == 0 or not raw:
            return 0.0
        for length in range(n, 0, -1):
            if raw.find(pat[:length]) >= 0:
                return float(length) / float(n)
        return 0.0

    def _write_terrain(self, text: str, windows: int = 4) -> None:
        """Project the example into the gap field as exogenous terrain.

        The organisms never see the text; they only feel it through the medium,
        exactly as they feel each other's emissions.
        """
        raw = text.encode("utf-8", errors="ignore")
        if not raw:
            return
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        parity = np.arange(ids.size, dtype=np.int32) & 1
        token_ids = np.clip(ids + 256 * parity, 0, self.base_latent.shape[0] - 1)
        chunks = np.array_split(token_ids, max(1, int(windows)))
        for w, chunk in enumerate(chunks):
            if chunk.size == 0:
                continue
            vec = _normalize(self.base_latent[chunk].mean(axis=0))
            point = np.tanh(vec[: self.cfg.gap_dim]).astype(np.float32)
            self.gap.write(
                point=point,
                velocity=np.zeros((self.cfg.gap_dim,), dtype=np.float32),
                phase=float(np.arctan2(point[1], point[0])),
                omega=float(self.cfg.omega_base),
                energy=1.0,
                genome_fragment=chunk[: self.cfg.proposal_lmax] & 0xFF,
                genome_weight=1.0,
                ifs_fragment=None,
                ifs_weight=0.0,
                emitter_id=-1,
                step_idx=self.step_idx,
                round_idx=w,
            )

    def _resonance_latent(self, gap_vec: np.ndarray) -> np.ndarray:
        out = np.zeros((self.cfg.d_latent,), dtype=np.float32)
        d = min(self.cfg.gap_dim, self.cfg.d_latent)
        out[:d] = np.asarray(gap_vec, dtype=np.float32)[:d]
        return out

    # ------------------------------------------------------------- one round

    def _run_rounds(
        self, raw: bytes, feature: dict[int, float], charge: bool
    ) -> tuple[dict[int, float], dict[int, float]]:
        """K synchronised chaos rounds over the organisms this example concerns.

        Returns the best wake score each one reached and its last resonance.
        """
        ccfg = self.ccfg
        ids = [tid for tid in feature if tid in self.population]
        wake: dict[int, float] = {tid: 0.0 for tid in ids}
        strength: dict[int, float] = {tid: 0.0 for tid in ids}
        if not ids:
            return wake, strength

        batch = max(int(self.cfg.gap_read_batch_size), 1)
        d_gap = int(self.cfg.gap_dim)
        for round_idx in range(self.cfg.k_rounds):
            self.gap.decay()
            live = [
                tid
                for tid in ids
                if tid in self.population
                and self.population[tid].energy > ccfg.min_viable_energy
            ]
            if not live:
                break
            for i0 in range(0, len(live), batch):
                chunk = live[i0 : i0 + batch]
                sig = np.zeros((batch, d_gap), dtype=np.float32)
                width = np.zeros((batch,), dtype=np.float32)
                phase = np.zeros((batch,), dtype=np.float32)
                coupling = np.zeros((batch,), dtype=np.float32)
                valid = np.zeros((batch,), dtype=np.float32)
                for j, tid in enumerate(chunk):
                    v = self.population[tid]
                    sig[j] = v.signature
                    width[j] = v.resonance_width
                    phase[j] = v.phase
                    coupling[j] = v.phase_coupling
                    valid[j] = 1.0

                resonance, read_strength = self.gap.read_many(
                    receiver_signature=sig,
                    resonance_width=width,
                    receiver_phase=phase,
                    phase_coupling=coupling,
                    valid_mask=valid,
                )

                for j, tid in enumerate(chunk):
                    voter = self.population.get(tid)
                    if voter is None or voter.energy <= ccfg.min_viable_energy:
                        continue
                    rs = float(read_strength[j])
                    strength[tid] = rs
                    voter.chaos_step(
                        self.rng, self._resonance_latent(resonance[j])
                    )
                    score = (
                        ccfg.wake_feature_weight * feature.get(tid, 0.0)
                        + ccfg.wake_resonance_weight * rs
                    )
                    wake[tid] = max(wake[tid], score)
                    if score < voter.activation_threshold:
                        continue
                    if charge:
                        if voter.energy <= ccfg.emit_cost + ccfg.min_viable_energy:
                            continue
                        self._reservoir_add(self._drain(voter, ccfg.emit_cost))
                    emit, vel, _ = voter.emission(self.cfg, voter.state_vec[:2])
                    self.gap.write(
                        point=emit,
                        velocity=vel,
                        phase=float(voter.phase),
                        omega=float(voter.omega),
                        energy=float(max(voter.energy, 0.0)),
                        genome_fragment=voter.identity_bytes,
                        genome_weight=float(feature.get(tid, 0.0)),
                        ifs_fragment=voter.ifs,
                        ifs_weight=float(feature.get(tid, 0.0)),
                        emitter_id=int(tid),
                        step_idx=self.step_idx,
                        round_idx=round_idx,
                    )
        return wake, strength

    def _collect_votes(
        self, wake: dict[int, float], strength: dict[int, float], charge: bool
    ) -> tuple[list[Vote], list[int]]:
        ccfg = self.ccfg
        votes: list[Vote] = []
        abstained: list[int] = []
        for tid in wake:
            voter = self.population.get(tid)
            if voter is None:
                continue
            if wake[tid] < voter.activation_threshold:
                abstained.append(tid)
                continue
            query = voter.query()
            scores = self.label_latent @ query
            order = np.argsort(scores)[::-1]
            best = int(order[0])
            margin = float(scores[order[0]] - scores[order[1]]) if scores.size > 1 else 1.0
            conf = float(1.0 / (1.0 + np.exp(-ccfg.vote_margin_scale * margin)))
            conf *= 0.5 + 0.5 * float(np.clip(wake.get(tid, 0.0), 0.0, 1.0))

            stake = voter.stake_scale * conf * voter.energy * ccfg.stake_max_energy_frac
            stake = float(min(max(stake, ccfg.stake_min), voter.energy * ccfg.stake_max_energy_frac))
            if stake <= 0.0 or voter.energy <= ccfg.min_viable_energy:
                abstained.append(tid)
                continue
            if charge:
                stake = self._drain(voter, stake)
                if stake <= 0.0:
                    abstained.append(tid)
                    continue
            votes.append(Vote(token_id=tid, label=best, confidence=conf, stake=stake))
        return votes, abstained

    def _readout(self, votes: Sequence[Vote]) -> tuple[int, np.ndarray]:
        ccfg = self.ccfg
        scores = np.zeros((self.num_labels,), dtype=np.float64)
        for vote in votes:
            voter = self.population.get(vote.token_id)
            if voter is None:
                continue
            roi = (voter.won_total + ccfg.roi_prior) / (
                voter.staked_total + ccfg.roi_prior
            )
            weight = (
                vote.confidence
                * float(max(voter.energy, 0.0) + 1e-6)
                * float(np.clip(roi, 0.0, ccfg.roi_max))
            )
            scores[vote.label] += weight
        if float(scores.sum()) <= 0.0:
            probs = np.full((self.num_labels,), 1.0 / self.num_labels)
            return int(self.rng.integers(0, self.num_labels)), probs
        logits = scores / max(float(scores.max()), 1e-9)
        exp = np.exp(logits / self.ccfg.readout_temperature)
        probs = exp / float(exp.sum())
        return int(np.argmax(scores)), probs

    # -------------------------------------------------------------- training

    def train_step(self, example: Example) -> StepReport:
        self.step_idx += 1
        ccfg = self.ccfg
        raw = example.text.encode("utf-8", errors="ignore")

        self._write_terrain(example.text)
        feature = self._candidates(raw)
        wake, strength = self._run_rounds(raw, feature, charge=True)
        votes, abstained = self._collect_votes(wake, strength, charge=True)
        predicted, _ = self._readout(votes)

        pot = float(sum(v.stake for v in votes))
        rake = pot * ccfg.rake_frac
        payout_pool = pot - rake
        self._reservoir_add(rake)

        winners = [v for v in votes if v.label == example.label]
        win_stake = float(sum(v.stake for v in winners))
        if winners and win_stake > 0.0:
            for vote in winners:
                voter = self.population.get(vote.token_id)
                if voter is None:
                    continue
                share = payout_pool * (vote.stake / win_stake)
                self._credit(voter, share)
                voter.wins += 1
                voter.won_total += share
        else:
            self._reservoir_add(payout_pool)

        for vote in votes:
            voter = self.population.get(vote.token_id)
            if voter is None:
                continue
            voter.votes += 1
            voter.staked_total += vote.stake
            voter.inactivity_steps = 0
            voter.last_active_step = self.step_idx
            if vote.label != example.label:
                voter.losses += 1

        if self.step_idx % ccfg.economy_interval == 0:
            self._run_economy(ccfg.economy_interval)

        births = self._reproduce(
            example,
            correct=bool(winners),
            losers=[v for v in votes if v.label != example.label],
            winners=winners,
        )
        deaths = self._reap()

        correct = predicted == example.label
        self.recent_correct.append(1 if correct else 0)
        if len(self.recent_correct) > 500:
            self.recent_correct.pop(0)

        return StepReport(
            step=self.step_idx,
            correct=correct,
            predicted=predicted,
            truth=example.label,
            voters=len(votes),
            abstained=len(self.population) - len(votes),
            pot=pot,
            winners=len(winners),
            births=births,
            deaths=deaths,
            population=len(self.population),
            reservoir=self.reservoir,
            total_energy=self._total_energy(),
        )

    def _run_economy(self, interval: int) -> None:
        """Charge metabolism and pay silence credit for the last `interval` steps.

        Existing costs energy whether or not you act; abstaining costs nothing
        and earns a subsistence rebate while you are poor. Compounding the tax
        over the interval in closed form keeps this identical to charging it
        every step.
        """
        ccfg = self.ccfg
        decay = 1.0 - (1.0 - ccfg.metabolic_tax_rate) ** max(int(interval), 1)
        for voter in self.population.values():
            self._reservoir_add(self._drain(voter, voter.energy * decay))
            idle = self.step_idx - voter.last_active_step
            voter.inactivity_steps = int(max(idle, 0))
            if idle >= interval and voter.energy < ccfg.silence_energy_ceiling:
                credit = self._silence_credit(voter) * float(interval)
                self._credit(voter, self._reservoir_take(credit))

    def _reap(self) -> int:
        ccfg = self.ccfg
        dead = {
            tid
            for tid, v in self.population.items()
            if v.energy <= ccfg.min_viable_energy
        }
        surplus = len(self.population) - len(dead) - ccfg.max_population
        if surplus > 0:
            # Over the cap the poorest go first, but organisms still inside
            # their grace window are spared: a newborn has not had the chance
            # to meet an example its pattern fits.
            grown = [
                tid
                for tid, v in self.population.items()
                if tid not in dead
                and self.step_idx - v.birth_step >= ccfg.newborn_grace_steps
            ]
            grown.sort(key=lambda t: float(self.population[t].energy))
            dead.update(grown[:surplus])
        for tid in dead:
            voter = self.population.get(tid)
            if voter is None:
                continue
            self._reservoir_add(max(voter.energy, 0.0))
            self._expel(voter)
        self.gap.purge_emitters(dead)
        self.deaths += len(dead)
        return len(dead)

    def _mutate_from(self, parent: Voter, energy: float) -> Voter:
        ccfg = self.ccfg
        child = self._make_voter(
            pattern=parent.identity_bytes,
            label_lean=None,
            energy=energy,
            parent_a=parent.token_id,
        )
        factor = _sample_pareto_factor(
            self.rng, ccfg.pareto_alpha, ccfg.pareto_scale, ccfg.pareto_clip
        )
        child.identity_bytes = _mutate_pattern(
            parent.identity_bytes, self.rng, ccfg
        )
        child.refresh_pattern()
        child.vote_bias = _normalize(
            parent.vote_bias
            + self.rng.normal(
                0.0, ccfg.vote_mutation_scale, size=parent.vote_bias.shape
            ).astype(np.float32)
        ).astype(np.float32)
        child.ifs = np.clip(
            parent.ifs
            + self.rng.normal(0.0, self.cfg.ifs_mutation_scale, size=parent.ifs.shape),
            -4.0,
            4.0,
        ).astype(np.float32)
        child.activation_threshold = _mutate_scalar(
            parent.activation_threshold * factor, ccfg.mutation_scale, 0.02, 0.98, self.rng
        )
        child.resonance_width = _mutate_scalar(
            parent.resonance_width, ccfg.mutation_scale, 0.05, 3.0, self.rng
        )
        child.emission_amplitude = _mutate_scalar(
            parent.emission_amplitude, ccfg.mutation_scale, 0.02, 2.0, self.rng
        )
        child.silence_growth_rate = _mutate_scalar(
            parent.silence_growth_rate, ccfg.mutation_scale, 0.05, 6.0, self.rng
        )
        child.context_gain = _mutate_scalar(
            parent.context_gain, ccfg.mutation_scale, 0.01, 3.0, self.rng
        )
        child.stake_scale = _mutate_scalar(
            parent.stake_scale, ccfg.mutation_scale, 0.05, 4.0, self.rng
        )
        return child

    def _reproduce(
        self,
        example: Example,
        correct: bool,
        losers: Sequence[Vote] = (),
        winners: Sequence[Vote] = (),
    ) -> int:
        ccfg = self.ccfg
        births = 0

        # Discovery: nobody staked the right label, so the truth has to be paid
        # for. The system buys an organism that recognises this example. Half
        # the time it reuses the pattern of whichever organism staked most on
        # the wrong answer — that pattern demonstrably fires here, so only its
        # label lean was wrong. This is the classifier's version of the engine
        # refining a proposer from revealed ground truth.
        if not correct and self.rng.random() < ccfg.discovery_spawn_prob:
            ranked = sorted(losers, key=lambda v: float(v.stake), reverse=True)
            for i in range(ccfg.discovery_spawn_max_per_step):
                energy = self._reservoir_take(ccfg.spawn_energy)
                if energy <= ccfg.min_viable_energy:
                    break
                pattern = None
                if i < len(ranked) and self.rng.random() < ccfg.discovery_reuse_prob:
                    parent = self.population.get(ranked[i].token_id)
                    if parent is not None:
                        pattern = parent.identity_bytes.copy()
                if pattern is None:
                    pattern = _sample_pattern(example.text, self.rng, ccfg)
                child = self._make_voter(pattern, label_lean=example.label, energy=energy)
                self._admit(child)
                births += 1

        # Success breeds, but only for organisms that just won and can afford
        # it. Selecting on current winners instead of on global wealth keeps
        # the same few lineages from flooding the population with duplicates.
        candidates = []
        for vote in winners:
            parent = self.population.get(vote.token_id)
            if parent is not None and parent.energy >= ccfg.self_copy_min_energy:
                candidates.append(parent)
        candidates.sort(key=lambda v: float(v.energy), reverse=True)
        for parent in candidates[: ccfg.self_copy_max_per_step]:
            paid = self._drain(parent, ccfg.self_copy_cost)
            if paid <= 0.0:
                continue
            child = self._mutate_from(parent, energy=paid)
            self._admit(child)
            births += 1

        self.births += births
        return births

    # ------------------------------------------------------------- inference

    def predict(self, text: str) -> tuple[int, np.ndarray, int]:
        """Typed decision for one example: (label, probabilities, voters).

        No energy changes hands, so evaluation cannot perturb the population.
        """
        raw = text.encode("utf-8", errors="ignore")
        snapshot = {
            tid: (v.state_vec.copy(), float(v.phase), int(v.inactivity_steps))
            for tid, v in self.population.items()
        }
        gap_state = (
            self.gap.points.copy(),
            self.gap.velocity.copy(),
            self.gap.phase.copy(),
            self.gap.omega.copy(),
            self.gap.energy.copy(),
            self.gap.emitter_id.copy(),
            int(self.gap.ptr),
        )
        try:
            self._write_terrain(text)
            feature = self._candidates(raw)
            wake, strength = self._run_rounds(raw, feature, charge=False)
            votes, _ = self._collect_votes(wake, strength, charge=False)
            label, probs = self._readout(votes)
            return label, probs, len(votes)
        finally:
            for tid, (state, phase, inactivity) in snapshot.items():
                voter = self.population.get(tid)
                if voter is None:
                    continue
                voter.state_vec = state
                voter.phase = phase
                voter.inactivity_steps = inactivity
            (
                self.gap.points,
                self.gap.velocity,
                self.gap.phase,
                self.gap.omega,
                self.gap.energy,
                self.gap.emitter_id,
                self.gap.ptr,
            ) = gap_state

    def evaluate(self, examples: Sequence[Example]) -> dict:
        if not examples:
            return {"accuracy": 0.0, "n": 0}
        hits = 0
        voter_counts: list[int] = []
        confusion = np.zeros((self.num_labels, self.num_labels), dtype=np.int64)
        nll = 0.0
        for ex in examples:
            label, probs, n_voters = self.predict(ex.text)
            confusion[ex.label, label] += 1
            voter_counts.append(n_voters)
            nll -= float(np.log(max(probs[ex.label], 1e-9)))
            if label == ex.label:
                hits += 1
        n = len(examples)
        per_class = np.zeros((self.num_labels,), dtype=np.float64)
        for c in range(self.num_labels):
            total = int(confusion[c].sum())
            per_class[c] = float(confusion[c, c]) / total if total else 0.0
        return {
            "accuracy": hits / n,
            "macro_accuracy": float(per_class.mean()),
            "mean_nll": nll / n,
            "mean_voters": float(np.mean(voter_counts)) if voter_counts else 0.0,
            "n": n,
            "confusion": confusion.tolist(),
            "per_class": per_class.tolist(),
        }

    # ------------------------------------------------------------ inspection

    def describe_population(self, top: int = 20) -> list[str]:
        """The population as readable rules, richest first."""
        rows: list[str] = []
        ranked = sorted(
            self.population.values(), key=lambda v: float(v.energy), reverse=True
        )[: max(top, 0)]
        for v in ranked:
            lean = int(np.argmax(self.label_latent @ v.query()))
            pattern = _pattern_bytes(v.identity_bytes).decode("utf-8", errors="replace")
            plays = max(v.votes, 1)
            rows.append(
                f"  {pattern!r:<20} -> {self.task.labels[lean]:<10} "
                f"E={v.energy:6.2f} votes={v.votes:5d} win%={100.0 * v.wins / plays:5.1f} "
                f"thr={v.activation_threshold:.2f} stake={v.stake_scale:.2f} "
                f"age={self.step_idx - v.birth_step}"
            )
        return rows

    def stats(self) -> dict:
        energies = np.asarray(
            [v.energy for v in self.population.values()], dtype=np.float64
        )
        votes = np.asarray([v.votes for v in self.population.values()], dtype=np.float64)
        recent = (
            float(np.mean(self.recent_correct)) if self.recent_correct else 0.0
        )
        return {
            "step": self.step_idx,
            "population": len(self.population),
            "rolling_accuracy": recent,
            "reservoir": float(self.reservoir),
            "total_energy": self._total_energy(),
            "energy_drift": self._total_energy() - self.total_energy_ref,
            "mean_energy": float(energies.mean()) if energies.size else 0.0,
            "max_energy": float(energies.max()) if energies.size else 0.0,
            "mean_votes": float(votes.mean()) if votes.size else 0.0,
            "births": self.births,
            "deaths": self.deaths,
        }
