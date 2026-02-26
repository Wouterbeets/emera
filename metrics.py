from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math


@dataclass
class MetricsTracker:
    steps: int = 0
    cumulative_distance: int = 0
    cumulative_predicted_advance: int = 0
    cumulative_raw_predicted_advance: int = 0
    cumulative_contextual_predicted_advance: int = 0
    cumulative_proposed_tokens: int = 0
    cumulative_discovery_advance: int = 0
    cumulative_energy_spent: float = 0.0
    cumulative_net_return: float = 0.0
    cumulative_payout: float = 0.0
    cumulative_births: int = 0
    cumulative_deaths: int = 0
    cumulative_nontrivial_matches: int = 0
    cumulative_full_matches: int = 0
    max_symbio_depth: int = 0
    last_gap_compression_ratio: float = 1.0
    best_gap_compression_ratio: float = float("inf")
    last_root_only_fraction: float = 1.0
    last_events: List[str] = field(default_factory=list)

    def update(self, step_stats: Dict[str, float | int], events: list[str]) -> None:
        self.steps += 1
        adv = int(step_stats.get("advance_len", 0))
        match = int(step_stats.get("match_len", 0))
        p_len = int(step_stats.get("proposal_len", 0))
        rescue_winner = bool(int(step_stats.get("winner_from_frontier_rescue", 0)))
        match_for_glide = 0 if rescue_winner else match
        self.cumulative_distance += adv
        self.cumulative_raw_predicted_advance += match
        self.cumulative_predicted_advance += match_for_glide
        self.cumulative_contextual_predicted_advance += max(match_for_glide - 1, 0)
        self.cumulative_proposed_tokens += max(p_len, 0)
        self.cumulative_discovery_advance += int(step_stats.get("discovery_advance", 0))
        self.cumulative_energy_spent += float(step_stats.get("energy_spent", 0.0))
        self.cumulative_net_return += float(step_stats.get("realized_return", 0.0))
        self.cumulative_payout += float(step_stats.get("jackpot", 0.0))
        self.cumulative_births += int(step_stats.get("births", 0))
        self.cumulative_deaths += int(step_stats.get("deaths", 0))
        if match_for_glide >= 2:
            self.cumulative_nontrivial_matches += 1
        if p_len > 0 and match_for_glide == p_len:
            self.cumulative_full_matches += 1
        self.max_symbio_depth = max(self.max_symbio_depth, int(step_stats.get("max_symbio_depth", 0)))
        gcr = float(step_stats.get("gap_compression_ratio", self.last_gap_compression_ratio))
        if not math.isfinite(gcr):
            gcr = self.last_gap_compression_ratio
        self.last_gap_compression_ratio = gcr
        self.best_gap_compression_ratio = min(self.best_gap_compression_ratio, gcr)
        self.last_root_only_fraction = float(step_stats.get("root_only_fraction", self.last_root_only_fraction))
        if events:
            self.last_events.extend(events)
            self.last_events = self.last_events[-16:]

    def snapshot(self) -> dict:
        distance = max(float(self.cumulative_distance), 1.0)
        proposed = max(float(self.cumulative_proposed_tokens), 1.0)
        steps = max(float(self.steps), 1.0)
        return {
            "steps": self.steps,
            "distance": self.cumulative_distance,
            "predicted_advance": self.cumulative_predicted_advance,
            "raw_predicted_advance": self.cumulative_raw_predicted_advance,
            "contextual_predicted_advance": self.cumulative_contextual_predicted_advance,
            "proposed_tokens": self.cumulative_proposed_tokens,
            "discovery_advance": self.cumulative_discovery_advance,
            "energy_spent": self.cumulative_energy_spent,
            "net_return": self.cumulative_net_return,
            "payout_total": self.cumulative_payout,
            "energy_per_advance": self.cumulative_energy_spent / distance,
            # Strict glide: only counts predictive advance beyond trivial 1-token wins.
            "glide_ratio": self.cumulative_contextual_predicted_advance / distance,
            # Legacy glide for debugging/compatibility.
            "token_glide_ratio": self.cumulative_raw_predicted_advance / distance,
            "proposal_efficiency": self.cumulative_raw_predicted_advance / proposed,
            "contextual_efficiency": self.cumulative_contextual_predicted_advance / proposed,
            "avg_proposal_len": self.cumulative_proposed_tokens / steps,
            "nontrivial_match_rate": self.cumulative_nontrivial_matches / steps,
            "full_match_rate": self.cumulative_full_matches / steps,
            "discovery_fraction": self.cumulative_discovery_advance / distance,
            "births_total": self.cumulative_births,
            "deaths_total": self.cumulative_deaths,
            "max_symbio_depth": int(self.max_symbio_depth),
            "gap_compression_ratio": float(self.last_gap_compression_ratio),
            "best_gap_compression_ratio": float(
                self.best_gap_compression_ratio
                if math.isfinite(self.best_gap_compression_ratio)
                else self.last_gap_compression_ratio
            ),
            "root_only_fraction": float(self.last_root_only_fraction),
            "last_events": list(self.last_events[-8:]),
        }
