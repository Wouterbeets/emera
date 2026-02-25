from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricsTracker:
    steps: int = 0
    cumulative_distance: int = 0
    cumulative_predicted_advance: int = 0
    cumulative_discovery_advance: int = 0
    cumulative_energy_spent: float = 0.0
    cumulative_net_return: float = 0.0
    cumulative_payout: float = 0.0
    cumulative_births: int = 0
    cumulative_deaths: int = 0
    last_events: List[str] = field(default_factory=list)

    def update(self, step_stats: Dict[str, float | int], events: list[str]) -> None:
        self.steps += 1
        self.cumulative_distance += int(step_stats.get("advance_len", 0))
        self.cumulative_predicted_advance += int(step_stats.get("match_len", 0))
        self.cumulative_discovery_advance += int(step_stats.get("discovery_advance", 0))
        self.cumulative_energy_spent += float(step_stats.get("energy_spent", 0.0))
        self.cumulative_net_return += float(step_stats.get("realized_return", 0.0))
        self.cumulative_payout += float(step_stats.get("jackpot", 0.0))
        self.cumulative_births += int(step_stats.get("births", 0))
        self.cumulative_deaths += int(step_stats.get("deaths", 0))
        if events:
            self.last_events.extend(events)
            self.last_events = self.last_events[-16:]

    def snapshot(self) -> dict:
        distance = max(float(self.cumulative_distance), 1.0)
        return {
            "steps": self.steps,
            "distance": self.cumulative_distance,
            "predicted_advance": self.cumulative_predicted_advance,
            "discovery_advance": self.cumulative_discovery_advance,
            "energy_spent": self.cumulative_energy_spent,
            "net_return": self.cumulative_net_return,
            "payout_total": self.cumulative_payout,
            "energy_per_advance": self.cumulative_energy_spent / distance,
            "glide_ratio": self.cumulative_predicted_advance / distance,
            "discovery_fraction": self.cumulative_discovery_advance / distance,
            "births_total": self.cumulative_births,
            "deaths_total": self.cumulative_deaths,
            "last_events": list(self.last_events[-8:]),
        }
