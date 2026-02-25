from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


def _pair_key(a: int, b: int) -> tuple[int, int]:
    if a <= b:
        return (a, b)
    return (b, a)


@dataclass
class CooperationStats:
    alpha: float
    ind_ema: Dict[int, float] = field(default_factory=dict)
    pair_ema: Dict[tuple[int, int], float] = field(default_factory=dict)
    pair_hits: Dict[tuple[int, int], int] = field(default_factory=dict)

    def update(self, contributions: Dict[int, float], realized_return: float) -> None:
        if not contributions:
            return
        a = float(max(min(self.alpha, 1.0), 1e-6))
        for tid, c in contributions.items():
            prev = self.ind_ema.get(tid, 0.0)
            x = float(c) * float(realized_return)
            self.ind_ema[tid] = (1.0 - a) * prev + a * x

        ids = sorted(contributions.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ti, tj = ids[i], ids[j]
                key = _pair_key(ti, tj)
                mass = min(float(contributions[ti]), float(contributions[tj]))
                prev = self.pair_ema.get(key, 0.0)
                x = mass * float(realized_return)
                self.pair_ema[key] = (1.0 - a) * prev + a * x
                self.pair_hits[key] = self.pair_hits.get(key, 0) + 1

    def synergy(self, a: int, b: int) -> float:
        key = _pair_key(a, b)
        pair = self.pair_ema.get(key, 0.0)
        ind = 0.5 * (self.ind_ema.get(a, 0.0) + self.ind_ema.get(b, 0.0))
        return float(pair - ind)

    def top_pairs(self, min_hits: int = 1) -> list[tuple[int, int, float, int]]:
        out: list[tuple[int, int, float, int]] = []
        for (a, b), p in self.pair_ema.items():
            hits = self.pair_hits.get((a, b), 0)
            if hits < min_hits:
                continue
            s = p - 0.5 * (self.ind_ema.get(a, 0.0) + self.ind_ema.get(b, 0.0))
            if s > 0.0:
                out.append((a, b, float(s), int(hits)))
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def prune_dead(self, alive_ids: Iterable[int]) -> None:
        alive = set(int(x) for x in alive_ids)
        self.ind_ema = {k: v for k, v in self.ind_ema.items() if k in alive}
        self.pair_ema = {k: v for k, v in self.pair_ema.items() if (k[0] in alive and k[1] in alive)}
        self.pair_hits = {k: v for k, v in self.pair_hits.items() if (k[0] in alive and k[1] in alive)}
