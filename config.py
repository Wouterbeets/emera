from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class EmeraConfig:
    # Reproducibility
    seed: int = 42

    # Token space
    token_space: str = "byte_parity"  # {"byte_parity", "gpt2"}
    gpt2_model_name: str = "gpt2"

    # Vocabulary/world
    base_tokens: int = 512
    world_vocab_size: int = 128
    corpus_file: Optional[str] = None
    world_len: int = 200_000
    zipf_alpha: float = 1.15

    # Latent/gap geometry
    d_latent: int = 32
    gap_dim: int = 16
    gap_len: int = 128
    gap_decay: float = 0.97
    gap_velocity_decay: float = 0.90
    omega_base: float = 0.035
    omega_jitter: float = 0.010
    beacon_strength: float = 0.10

    # Chaos/proposal
    num_ifs: int = 4
    chaos_substeps_per_round: int = 1
    chaos_svg_default_steps: int = 2000
    chaos_svg_max_steps: int = 50000
    k_rounds: int = 6
    proposal_lmax: int = 8
    confidence_scale: float = 10.0
    proposal_drift_scale: float = 0.08
    obligatory_proposals: bool = True
    proposal_min_bet: float = 0.0005
    proposal_bet_unit_scale: float = 0.05
    proposal_bet_max_energy_frac: float = 0.12
    proposal_bet_floor_frac: float = 0.08
    proposal_bet_conf_gain: float = 8.0
    proposal_frontier_contrast: int = 8
    proposal_frontier_fallback: int = 24
    frontier_rescue_energy: float = 0.60
    frontier_rescue_noise: float = 0.04
    frontier_rescue_max_per_step: int = 1
    contrastive_enabled: bool = True
    contrastive_correct_reward: float = 0.06
    contrastive_wrong_penalty: float = 0.45
    contrastive_wrong_exp: float = 2.0

    # Initial super-token population
    initial_super_tokens: int = 100
    initial_super_energy: float = 3.00

    # Genome trait template
    activation_threshold_init: float = 0.45
    emission_amplitude_init: float = 0.40
    emission_decay_init: float = 0.15
    silence_growth_init: float = 1.00
    resonance_width_init: float = 0.45
    phase_coupling_init: float = 0.25
    velocity_coupling_init: float = 0.20
    proposal_length_bias_init: float = 0.00

    # Thermodynamic ledger
    ambient_dissipation: float = 0.003
    metabolic_tax_rate: float = 0.0015
    base_toll: float = 0.005
    attempt_cost_base: float = 0.04
    discovery_cost: float = 0.18
    token_energy_cap: float = 50.0
    min_viable_energy: float = 1e-4
    survivor_grace_steps: int = 8
    survivor_relief_active_frac: float = 0.35
    survivor_relief_reservoir_frac: float = 0.01
    strict_energy_budget: bool = True
    conserve_total_energy: bool = True
    energy_inflow_per_step: float = 0.0
    energy_reservoir_init: float = 120.0
    energy_reservoir_cap: float = 1000.0
    death_recycle_fraction: float = 1.0
    death_recycle_flat: float = 0.0

    # Silence credit
    silence_log_coeff: float = 0.004
    silence_exp_coeff: float = 0.0010
    silence_exp_rate: float = 0.015

    # Jackpot
    jackpot_base: float = 4.00
    jackpot_length_scale: float = 0.35
    jackpot_silence_scale: float = 0.40

    # Symbiogenesis
    spawn_cost: float = 1.00
    mint_interval: int = 25
    mint_delta: float = 0.00
    self_copy_enabled: bool = False
    self_copy_interval: int = 5
    self_copy_cost: float = 0.50
    self_copy_min_energy: float = 1.20
    self_copy_min_match_frac: float = 1.00
    self_copy_min_score: float = 0.0
    self_copy_max_per_step: int = 1
    ema_alpha: float = 0.02
    mutation_scale: float = 0.08
    ifs_mutation_scale: float = 0.03

    # Pareto mutation dynamics
    pareto_alpha_init: float = 1.7
    pareto_alpha_min: float = 1.1
    pareto_alpha_max: float = 4.5
    pareto_mutation_scale: float = 0.22
    pareto_clip: float = 12.0

    # Reward split for child success
    child_reward_share: float = 0.80
    parent_reward_share: float = 0.10

    # Adaptive natural laws (environment-driven)
    dynamic_laws: bool = True
    law_update_interval: int = 25
    adaptation_ema_decay: float = 0.97
    adaptation_rate: float = 0.05
    adaptation_signal_decay: float = 0.70
    target_active_super: float = 100.0
    target_match_rate: float = 0.08
    target_proposal_pressure: float = 0.70
    target_birth_death_gap: float = 0.0

    # Law bounds
    attempt_cost_min: float = 0.005
    attempt_cost_max: float = 0.40
    jackpot_base_min: float = 0.20
    jackpot_base_max: float = 20.0
    silence_log_min: float = 1e-5
    silence_log_max: float = 0.04
    silence_exp_min: float = 1e-6
    silence_exp_max: float = 0.02
    ambient_dissipation_min: float = 1e-4
    ambient_dissipation_max: float = 0.08
    spawn_cost_min: float = 0.05
    spawn_cost_max: float = 8.0
    mint_delta_min: float = -0.30
    mint_delta_max: float = 0.60

    # Seasonal forcing
    seasons_enabled: bool = True
    season_period: int = 400
    season_strength: float = 0.30
    season_wave_decay: float = 0.65
    season_revival_spores: int = 24
    season_revival_energy: float = 1.1

    # Logging
    log_every: int = 50

    def validate(self) -> None:
        if self.token_space not in {"byte_parity", "gpt2"}:
            raise ValueError("token_space must be one of: {'byte_parity', 'gpt2'}.")
        if self.token_space == "byte_parity" and self.base_tokens != 512:
            raise ValueError("byte_parity mode expects base_tokens == 512.")
        if self.base_tokens < 8:
            raise ValueError("base_tokens must be >= 8.")
        if self.world_vocab_size < 8 or self.world_vocab_size > self.base_tokens:
            raise ValueError("world_vocab_size must be in [8, base_tokens].")
        if self.d_latent < 4:
            raise ValueError("d_latent must be >= 4.")
        if self.gap_dim < 4:
            raise ValueError("gap_dim must be >= 4.")
        if self.gap_dim > self.d_latent:
            raise ValueError("gap_dim must be <= d_latent.")
        if self.num_ifs < 1:
            raise ValueError("num_ifs must be >= 1.")
        if self.chaos_substeps_per_round < 1:
            raise ValueError("chaos_substeps_per_round must be >= 1.")
        if self.chaos_svg_default_steps < 1:
            raise ValueError("chaos_svg_default_steps must be >= 1.")
        if self.chaos_svg_max_steps < self.chaos_svg_default_steps:
            raise ValueError("chaos_svg_max_steps must be >= chaos_svg_default_steps.")
        if self.k_rounds < 1:
            raise ValueError("k_rounds must be >= 1.")
        if self.proposal_lmax < 1:
            raise ValueError("proposal_lmax must be >= 1.")
        if self.proposal_min_bet < 0.0:
            raise ValueError("proposal_min_bet must be >= 0.")
        if self.proposal_bet_unit_scale <= 0.0:
            raise ValueError("proposal_bet_unit_scale must be > 0.")
        if not (0.0 < self.proposal_bet_max_energy_frac <= 1.0):
            raise ValueError("proposal_bet_max_energy_frac must be in (0, 1].")
        if not (0.0 <= self.proposal_bet_floor_frac <= 1.0):
            raise ValueError("proposal_bet_floor_frac must be in [0, 1].")
        if self.proposal_bet_conf_gain <= 0.0:
            raise ValueError("proposal_bet_conf_gain must be > 0.")
        if self.proposal_frontier_contrast < 0:
            raise ValueError("proposal_frontier_contrast must be >= 0.")
        if self.proposal_frontier_fallback < 0:
            raise ValueError("proposal_frontier_fallback must be >= 0.")
        if self.frontier_rescue_energy <= 0.0:
            raise ValueError("frontier_rescue_energy must be > 0.")
        if self.frontier_rescue_noise < 0.0:
            raise ValueError("frontier_rescue_noise must be >= 0.")
        if self.frontier_rescue_max_per_step < 0:
            raise ValueError("frontier_rescue_max_per_step must be >= 0.")
        if self.contrastive_correct_reward < 0.0:
            raise ValueError("contrastive_correct_reward must be >= 0.")
        if self.contrastive_wrong_penalty < 0.0:
            raise ValueError("contrastive_wrong_penalty must be >= 0.")
        if self.contrastive_wrong_exp < 1.0:
            raise ValueError("contrastive_wrong_exp must be >= 1.")
        if self.initial_super_tokens < 1:
            raise ValueError("initial_super_tokens must be >= 1.")
        if self.spawn_cost <= 0:
            raise ValueError("spawn_cost must be > 0.")
        if self.self_copy_interval < 1:
            raise ValueError("self_copy_interval must be >= 1.")
        if self.self_copy_cost <= 0.0:
            raise ValueError("self_copy_cost must be > 0.")
        if self.self_copy_min_energy < 0.0:
            raise ValueError("self_copy_min_energy must be >= 0.")
        if not (0.0 <= self.self_copy_min_match_frac <= 1.0):
            raise ValueError("self_copy_min_match_frac must be in [0, 1].")
        if self.self_copy_max_per_step < 0:
            raise ValueError("self_copy_max_per_step must be >= 0.")
        if self.token_energy_cap <= 0:
            raise ValueError("token_energy_cap must be > 0.")
        if self.min_viable_energy < 0.0:
            raise ValueError("min_viable_energy must be >= 0.")
        if self.survivor_grace_steps < 0:
            raise ValueError("survivor_grace_steps must be >= 0.")
        if not (0.0 <= self.survivor_relief_active_frac <= 1.0):
            raise ValueError("survivor_relief_active_frac must be in [0, 1].")
        if not (0.0 <= self.survivor_relief_reservoir_frac <= 1.0):
            raise ValueError("survivor_relief_reservoir_frac must be in [0, 1].")
        if self.metabolic_tax_rate < 0.0:
            raise ValueError("metabolic_tax_rate must be >= 0.")
        if self.energy_inflow_per_step < 0.0:
            raise ValueError("energy_inflow_per_step must be >= 0.")
        if self.energy_reservoir_init < 0.0:
            raise ValueError("energy_reservoir_init must be >= 0.")
        if self.energy_reservoir_cap <= 0.0:
            raise ValueError("energy_reservoir_cap must be > 0.")
        if self.conserve_total_energy and self.energy_inflow_per_step != 0.0:
            raise ValueError("conserve_total_energy requires energy_inflow_per_step == 0.")
        if not (0.0 <= self.death_recycle_fraction <= 1.0):
            raise ValueError("death_recycle_fraction must be in [0, 1].")
        if self.death_recycle_flat < 0.0:
            raise ValueError("death_recycle_flat must be >= 0.")
        if self.mint_interval < 1:
            raise ValueError("mint_interval must be >= 1.")
        if self.law_update_interval < 1:
            raise ValueError("law_update_interval must be >= 1.")
        if self.season_period < 2:
            raise ValueError("season_period must be >= 2.")
        if self.season_strength < 0.0:
            raise ValueError("season_strength must be >= 0.")
        if not (0.0 <= self.season_wave_decay < 1.0):
            raise ValueError("season_wave_decay must be in [0, 1).")
        if self.season_revival_spores < 0:
            raise ValueError("season_revival_spores must be >= 0.")
        if self.season_revival_energy <= 0.0:
            raise ValueError("season_revival_energy must be > 0.")
        if not (0.0 <= self.adaptation_ema_decay < 1.0):
            raise ValueError("adaptation_ema_decay must be in [0, 1).")
        if self.adaptation_rate < 0.0:
            raise ValueError("adaptation_rate must be >= 0.")
        if not (0.0 <= self.adaptation_signal_decay < 1.0):
            raise ValueError("adaptation_signal_decay must be in [0, 1).")
        if self.pareto_alpha_init < 1.01:
            raise ValueError("pareto_alpha_init must be >= 1.01.")
        if self.pareto_alpha_min < 1.01 or self.pareto_alpha_max < self.pareto_alpha_min:
            raise ValueError("invalid pareto alpha bounds.")
        if not (0.0 <= self.child_reward_share <= 1.0):
            raise ValueError("child_reward_share must be in [0, 1].")
        if not (0.0 <= self.parent_reward_share <= 1.0):
            raise ValueError("parent_reward_share must be in [0, 1].")
        total = self.child_reward_share + 2.0 * self.parent_reward_share
        if total > 1.0 + 1e-6:
            raise ValueError("child_reward_share + 2*parent_reward_share must be <= 1.")
        if self.corpus_file:
            path = Path(self.corpus_file)
            if not path.exists():
                raise FileNotFoundError(f"corpus_file does not exist: {path}")
