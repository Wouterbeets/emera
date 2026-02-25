from __future__ import annotations

import numpy as np

from config import EmeraConfig
from genome import SuperToken


def attempt_cost(token: SuperToken, cfg: EmeraConfig) -> float:
    return attempt_cost_with_base(token, cfg, cfg.attempt_cost_base)


def attempt_cost_with_base(token: SuperToken, cfg: EmeraConfig, base_cost: float) -> float:
    amp = max(float(token.emission_amplitude), 0.0)
    return float(base_cost * (1.0 + 0.25 * amp))


def discovery_cost(unmatched_len: int, cfg: EmeraConfig) -> float:
    return float(cfg.discovery_cost * max(int(unmatched_len), 0))


def base_toll(advance_len: int, cfg: EmeraConfig) -> float:
    return float(cfg.base_toll * max(int(advance_len), 0))


def silence_credit(token: SuperToken, cfg: EmeraConfig) -> float:
    return silence_credit_with_coeffs(token, cfg, cfg.silence_log_coeff, cfg.silence_exp_coeff)


def silence_credit_with_coeffs(
    token: SuperToken,
    cfg: EmeraConfig,
    log_coeff: float,
    exp_coeff: float,
) -> float:
    s = max(int(token.inactivity_steps), 0)
    g = max(float(token.silence_growth_rate), 1e-6)
    log_term = float(log_coeff) * g * np.log1p(float(s))
    expo = np.clip(cfg.silence_exp_rate * g * float(s), 0.0, 40.0)
    exp_term = float(exp_coeff) * (np.exp(expo) - 1.0)
    return float(max(log_term + exp_term, 0.0))


def jackpot_reward(
    cfg: EmeraConfig,
    match_len: int,
    proposal_len: int,
    quality: float,
    rarity: float,
    inactivity_steps: int,
) -> float:
    return jackpot_reward_with_base(
        cfg=cfg,
        jackpot_base_value=cfg.jackpot_base,
        match_len=match_len,
        proposal_len=proposal_len,
        quality=quality,
        rarity=rarity,
        inactivity_steps=inactivity_steps,
    )


def jackpot_reward_with_base(
    cfg: EmeraConfig,
    jackpot_base_value: float,
    match_len: int,
    proposal_len: int,
    quality: float,
    rarity: float,
    inactivity_steps: int,
) -> float:
    if match_len <= 0 or proposal_len <= 0:
        return 0.0
    length_factor = 1.0 + cfg.jackpot_length_scale * float(max(match_len - 1, 0))
    quality_factor = float(np.clip(quality, 0.0, 1.5))
    rarity_factor = float(np.clip(rarity, 0.0, 3.0))
    silence_mul = 1.0 + cfg.jackpot_silence_scale * np.log1p(float(max(inactivity_steps, 0)))
    return float(float(jackpot_base_value) * rarity_factor * length_factor * quality_factor * silence_mul)


def realized_return(
    jackpot: float,
    total_attempt_cost: float,
    discovery: float,
    base: float,
) -> float:
    return float(jackpot - total_attempt_cost - discovery - base)
