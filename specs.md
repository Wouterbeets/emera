# Emera v2 Specification

Version: 0.1  
Date: 2026-02-21  
Status: Implementation spec for first runnable prototype

## 1. Objective

Build a non-loss-based, evolutionary language system that traverses a fixed token stream ("book-world") while minimizing:

`energy_spent / distance_advanced`

Base UTF-8 tokens are the immutable ground truth. Super-tokens emerge by symbiogenesis when cooperation improves traversal economics.

## 2. Design Invariants

1. Strict emergence: no hard-coded roles (no orchestrator/specialist classes).
2. No centralized loss, no attention, no shared token weights.
3. Survival is thermodynamic: useful entities stay alive; useless ones die by energy depletion.
4. Parents remain after minting; children are additive.
5. Keep moving parts minimal for solo-dev iteration speed.

## 3. Practical Constraints

1. Single developer, limited time.
2. Single RTX 4090.
3. Must run fast enough for overnight experiments.
4. Prefer deterministic, replayable runs (fixed seeds).

## 4. Vocabulary and World

### 4.1 Base tokens

Use the same encoding as `emera_fractal.py`:

`token_id = byte + 256 * parity`

- `byte in [0..255]`
- `parity in {0,1}`
- Total base tokens: 512

### 4.2 Book-world

1. The world is a fixed token sequence.
2. Global cursor points to current frontier.
3. Ground truth next tokens are directly available only through discovery (expensive).

## 5. Initialization: Distinct Token Identity

Goal: make all 512 base tokens maximally distinguishable in latent/gap space at step 0.

### 5.1 Immutable identity code

1. Build `H = WalshHadamard(512)`.
2. Assign each token:
   - `id_code[token_id] = H[token_id] / sqrt(512)`  
   (exact orthogonality at init).

### 5.2 Latent projection

1. Choose `d_latent` (default 32; optional 64).
2. Sample fixed random orthonormal `P: (512, d_latent)`.
3. Set:
   - `z[token_id] = normalize(id_code[token_id] @ P)`.

### 5.3 Deterministic per-token geometric seed

From `z[token_id]`, initialize:

1. `attractor0 = z[:2]`
2. `phase0 = atan2(z[1], z[0])`
3. `omega0 = omega_base + epsilon * z[2]`
4. `ifs0` generated deterministically from chunks of `z`

### 5.4 Identity beacon

Every emission includes a small fixed signature:

`emit = semantic_emit + beacon_strength * z[:gap_dim]`

Default `beacon_strength = 0.1`.

This preserves recognizability without role logic.

### 5.5 Energy initialization

All base tokens start with equal base energy:

`E_base_init = constant` (optionally tiny deterministic jitter only for tie-breaks).

## 6. Entities and State

### 6.1 BaseToken (immutable)

1. `token_id`
2. `id_code` / `z`
3. `ifs_base`
4. Optional frequency stats (for rarity estimation)

### 6.2 SuperToken (evolvable, mortal)

1. `token_id` (>= 512)
2. `parent_a`, `parent_b` (or `-1` for none)
3. `energy`
4. `alive` (derived from energy > 0)
5. `inactivity_steps`
6. `attractor`, `phase`, `omega`
7. Genome traits:
   - `ifs`
   - `activation_threshold`
   - `emission_amplitude`
   - `emission_decay`
   - `silence_growth_rate`
   - `resonance_width`
   - `phase_coupling`
   - `velocity_coupling`
   - `proposal_length_bias`

All new children start from same template distribution + inherited blend + small mutation.

### 6.3 Gap field (circular buffer)

Per slot:

1. `point` (vector)
2. `velocity`
3. `phase`
4. `omega`
5. `energy`
6. `emitter_id`
7. `step_idx`, `round_idx`

## 7. Main Dynamics Per Global Step

Let current cursor be `t`.

1. Run `K` synchronized micro-rounds (default `K=6`).
2. In each round, each currently alive super-token:
   - reads gap field,
   - computes resonance,
   - performs one chaos update,
   - optionally emits into gap.
3. Gather prediction proposals from active super-tokens.
4. Select winner by expected net return.
5. Compare proposal to world tokens at cursor.
6. Apply energy ledger updates.
7. Advance cursor by matched/discovered length.
8. Update cooperation statistics from realized return.
9. Kill entities with `energy <= 0`.
10. Run minting check (no caps/cooldowns/age gates).

## 8. Prediction/Discovery Semantics

### 8.1 Proposal

Each active super-token proposes:

1. segment tokens `[y_1..y_L]` (`L <= L_max`)
2. confidence `q`
3. internal match-quality estimate

### 8.2 Evaluation

1. Compute longest prefix match with ground truth at cursor.
2. `match_len = m`
3. `quality` from confidence calibration and geometric coherence.

### 8.3 Cursor movement

1. If `m > 0`, cursor advances by `m`.
2. If mismatch, pay discovery and reveal unmatched ground-truth tokens.

## 9. Energy Ledger

At each step:

1. Base toll: charged for advancing time/travel.
2. Attempt cost: charged to active proposers.
3. Discovery cost: charged per unmatched revealed base token.
4. Silence credit: given to inactive super-tokens as function of inactivity streak.
5. Jackpot reward: applied on successful match.

### 9.1 Silence credit

Use simple accelerating curve:

`silence_credit_i = a_i * log(1 + inactivity_steps_i) + b_i * (exp(c_i * inactivity_steps_i) - 1)`

Parameters are evolvable via `silence_growth_rate`.

### 9.2 Jackpot

`jackpot = rarity_factor * length_factor * quality_factor * silence_multiplier`

where:

1. `rarity_factor` from online token/segment frequency (inverse log-frequency or online IDF),
2. `length_factor` grows with `match_len`,
3. `quality_factor` from confidence/match quality,
4. `silence_multiplier` tied to inactivity streak.

### 9.3 Realized step return

Define:

`R = jackpot - total_attempt_cost - discovery_cost - base_toll`

This is the canonical cooperation signal.

## 10. Cooperation Detection and Minting

Core principle: mint when pair cooperation yields more return together than alone.

### 10.1 Contribution attribution from gap provenance

For each event with return `R`:

1. Compute per-token influence contribution `c_i` from gap slots used in winning proposal (using `emitter_id` and resonance weighting).
2. Normalize contributions: `sum_i c_i = 1`.

### 10.2 Minimal running statistics

Maintain EMAs:

1. `ind_ema[i]`
2. `pair_ema[i,j]`
3. `pair_hits[i,j]`

Updates:

1. `ind_ema[i] <- (1-a) * ind_ema[i] + a * (c_i * R)`
2. `pair_ema[i,j] <- (1-a) * pair_ema[i,j] + a * (min(c_i, c_j) * R)`
3. `synergy[i,j] = pair_ema[i,j] - 0.5 * (ind_ema[i] + ind_ema[j])`

### 10.3 Mint criterion with ablation test

For top positive-synergy pairs, run one deterministic ablation:

1. `R_ab`: both active
2. `R_a`: mute `b`
3. `R_b`: mute `a`

Mint if:

1. `R_ab > max(R_a, R_b) + delta`
2. both parents can pay spawn energy

No extra guardrails:

1. no child cap
2. no cooldown
3. no age threshold

Children that are not useful will die.

### 10.4 Mint mechanics

1. Parents pay spawn cost from energy.
2. Child gets spawn energy and blended genome + mutation.
3. Parents remain alive.
4. Child positive rewards are split:
   - child keeps 80%
   - parent A gets 10%
   - parent B gets 10%
5. Child negative outcomes are paid by child.
6. Death rule: if child energy <= 0, child is removed.

This gives parents resilience through successful descendants while keeping mechanism minimal.

## 11. Death and Compaction

1. Any super-token with `energy <= 0` dies immediately.
2. Remove from active arrays.
3. Gap slots with dead emitter IDs are either:
   - retained as inert decaying history, or
   - zeroed on compaction (implementation choice; start with zeroing for simplicity).

No decomposition-to-parents in v1.

## 12. Minimal Module Layout (`emera/`)

1. `specs.md` (this file)
2. `config.py`
3. `world.py` (stream + rarity stats)
4. `identity.py` (Hadamard identity + projection init)
5. `gap_field.py`
6. `genome.py`
7. `ledger.py`
8. `cooperation.py` (attribution, EMAs, ablation, mint decision)
9. `engine.py` (global loop)
10. `metrics.py`
11. `run.py`

Single-process first; optimize only after behavior exists.

## 13. Default Hyperparameters (4090-friendly)

1. `d_latent = 32`
2. `gap_dim = 16`
3. `gap_len = 128`
4. `num_ifs = 4`
5. `K = 6`
6. `L_max = 4`
7. `base_toll = 0.01`
8. `attempt_cost = 0.1` (tune)
9. `discovery_cost = 0.4` per unmatched token (tune)
10. `spawn_cost = 1.0` total from both parents (tune)
11. `ema_alpha = 0.02`
12. `mint_delta = 0.02`

Start small, then increase complexity.

## 14. Core Metrics

Primary:

1. `energy_per_advance = cumulative_energy_spent / cumulative_distance`

Secondary:

1. `glide_ratio = predicted_advance / total_advance`
2. `discovery_fraction`
3. births/deaths per 1k steps
4. live super-token count
5. activation frequency distribution (expect heavy tail)
6. reward concentration (do few children dominate?)
7. synergy matrix sparsity and persistence

## 15. Prototype Milestones

### M0: Skeleton

1. Implement world, identity, base traversal, ledger without super-tokens.
2. Confirm deterministic replay.

### M1: Signalling + proposals

1. Add super-token chaos rounds and gap field.
2. Add prediction selection and return computation.

### M2: Cooperation + minting

1. Add attribution EMAs, synergy, ablation-based mint.
2. Add child reward split and death.

### M3: Emergence validation

1. Long runs on toy Zipf corpus.
2. Verify decreasing `energy_per_advance`.
3. Verify spontaneous behavioral stratification without role labels.

## 16. Non-Goals (for now)

1. No role taxonomies.
2. No hand-crafted guardrails for child counts/cooldowns/age.
3. No complex multi-agent resource market.
4. No fancy UI before core loop works.

## 17. First Implementation Task

Implement M0+M1 in one runnable loop with logging every N steps, then add M2 minting logic exactly as specified above.
