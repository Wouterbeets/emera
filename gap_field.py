from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from config import EmeraConfig


def _normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return v / n


@dataclass
class GapRead:
    resonance: np.ndarray
    strength: float
    contributions: Dict[int, float]


class GapField:
    def __init__(self, cfg: EmeraConfig):
        self.cfg = cfg
        self.points = np.zeros((cfg.gap_len, cfg.gap_dim), dtype=np.float32)
        self.velocity = np.zeros((cfg.gap_len, cfg.gap_dim), dtype=np.float32)
        self.phase = np.zeros((cfg.gap_len,), dtype=np.float32)
        self.omega = np.zeros((cfg.gap_len,), dtype=np.float32)
        self.energy = np.zeros((cfg.gap_len,), dtype=np.float32)
        # Heritable payload lane: what was written, not just where/when it was emitted.
        self.genome_fragment = np.full((cfg.gap_len, max(int(cfg.proposal_lmax), 1)), -1, dtype=np.int32)
        self.genome_len = np.zeros((cfg.gap_len,), dtype=np.int16)
        self.genome_weight = np.zeros((cfg.gap_len,), dtype=np.float32)
        self.ifs_fragment = np.zeros((cfg.gap_len, int(cfg.num_ifs), 2, 3), dtype=np.float32)
        self.ifs_weight = np.zeros((cfg.gap_len,), dtype=np.float32)
        self.emitter_id = np.full((cfg.gap_len,), -1, dtype=np.int64)
        self.step_idx = np.zeros((cfg.gap_len,), dtype=np.int64)
        self.round_idx = np.zeros((cfg.gap_len,), dtype=np.int32)
        self.ptr = 0

    def decay(self) -> None:
        self.energy *= self.cfg.gap_decay
        self.velocity *= self.cfg.gap_velocity_decay
        self.genome_weight *= self.cfg.gap_decay
        self.ifs_weight *= self.cfg.gap_decay
        self.phase = (self.phase + self.omega) % (2.0 * np.pi)

    def write(
        self,
        point: np.ndarray,
        velocity: np.ndarray,
        phase: float,
        omega: float,
        energy: float,
        genome_fragment: np.ndarray | None,
        genome_weight: float,
        ifs_fragment: np.ndarray | None,
        ifs_weight: float,
        emitter_id: int,
        step_idx: int,
        round_idx: int,
    ) -> None:
        p = self.ptr
        self.points[p] = point.astype(np.float32, copy=False)
        self.velocity[p] = velocity.astype(np.float32, copy=False)
        self.phase[p] = float(phase)
        self.omega[p] = float(omega)
        self.energy[p] = max(float(energy), 0.0)
        self.genome_fragment[p].fill(-1)
        if genome_fragment is None:
            self.genome_len[p] = 0
            self.genome_weight[p] = 0.0
        else:
            frag = np.asarray(genome_fragment, dtype=np.int32).reshape(-1)
            n = min(int(frag.size), int(self.genome_fragment.shape[1]))
            if n > 0:
                vmax = 255 if self.cfg.token_space == "byte_parity" else max(int(self.cfg.base_tokens) - 1, 0)
                self.genome_fragment[p, :n] = np.clip(frag[:n], 0, vmax).astype(np.int32, copy=False)
                self.genome_len[p] = int(n)
                self.genome_weight[p] = max(float(genome_weight), 0.0)
            else:
                self.genome_len[p] = 0
                self.genome_weight[p] = 0.0
        if ifs_fragment is None:
            self.ifs_fragment[p].fill(0.0)
            self.ifs_weight[p] = 0.0
        else:
            raw = np.asarray(ifs_fragment, dtype=np.float32).reshape(-1)
            need = int(self.cfg.num_ifs) * 6
            packed = np.zeros((need,), dtype=np.float32)
            n = min(int(raw.size), need)
            if n > 0:
                packed[:n] = raw[:n]
            frag = packed.reshape(int(self.cfg.num_ifs), 2, 3)
            self.ifs_fragment[p] = np.clip(frag, -4.0, 4.0).astype(np.float32, copy=False)
            self.ifs_weight[p] = max(float(ifs_weight), 0.0)
        self.emitter_id[p] = int(emitter_id)
        self.step_idx[p] = int(step_idx)
        self.round_idx[p] = int(round_idx)
        self.ptr = (p + 1) % self.cfg.gap_len

    def _weights(
        self,
        receiver_signature: np.ndarray,
        resonance_width: float,
        receiver_phase: float,
        phase_coupling: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        active = self.energy > 1e-9
        if not np.any(active):
            return active, np.zeros((0,), dtype=np.float32)
        pts = self.points[active]
        recv = _normalize_vec(receiver_signature)
        recv = np.nan_to_num(recv, nan=0.0, posinf=0.0, neginf=0.0)
        pts_norm = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-8)
        pts_norm = np.nan_to_num(pts_norm, nan=0.0, posinf=0.0, neginf=0.0)
        sim = np.clip(pts_norm @ recv, -1.0, 1.0)

        width = max(float(resonance_width), 1e-3)
        geo = np.exp(-((1.0 - sim) ** 2) / (2.0 * width * width))

        phase_align = 0.5 * (1.0 + np.cos(self.phase[active] - float(receiver_phase)))
        phase_gate = (1.0 - float(phase_coupling)) + float(phase_coupling) * phase_align

        w = self.energy[active] * geo.astype(np.float32) * phase_gate.astype(np.float32)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return active, w.astype(np.float32)

    def read(
        self,
        receiver_signature: np.ndarray,
        resonance_width: float,
        receiver_phase: float,
        phase_coupling: float,
    ) -> GapRead:
        active, w = self._weights(
            receiver_signature=receiver_signature,
            resonance_width=resonance_width,
            receiver_phase=receiver_phase,
            phase_coupling=phase_coupling,
        )
        if w.size == 0 or float(np.sum(w)) <= 1e-9:
            return GapRead(
                resonance=np.zeros((self.cfg.gap_dim,), dtype=np.float32),
                strength=0.0,
                contributions={},
            )
        pts = self.points[active]
        total_w = float(np.sum(w))
        resonance = np.sum(pts * w[:, None], axis=0) / max(total_w, 1e-9)

        active_energy = self.energy[active]
        strength = float(np.sum(w) / max(np.sum(active_energy), 1e-9))
        strength = float(np.clip(strength, 0.0, 1.0))

        emit = self.emitter_id[active]
        contrib_raw: Dict[int, float] = {}
        for eid, wi in zip(emit.tolist(), w.tolist()):
            if eid < 0:
                continue
            contrib_raw[eid] = contrib_raw.get(eid, 0.0) + float(wi)
        total = sum(contrib_raw.values())
        if total > 1e-9:
            contrib = {k: v / total for k, v in contrib_raw.items()}
        else:
            contrib = {}
        return GapRead(
            resonance=resonance.astype(np.float32),
            strength=strength,
            contributions=contrib,
        )

    def contribution_weights(
        self,
        receiver_signature: np.ndarray,
        resonance_width: float,
        receiver_phase: float,
        phase_coupling: float,
    ) -> Dict[int, float]:
        return self.read(
            receiver_signature=receiver_signature,
            resonance_width=resonance_width,
            receiver_phase=receiver_phase,
            phase_coupling=phase_coupling,
        ).contributions

    def purge_emitters(self, dead_emitters: Iterable[int]) -> None:
        dead = set(int(x) for x in dead_emitters)
        if not dead:
            return
        mask = np.isin(self.emitter_id, np.fromiter(dead, dtype=np.int64))
        if not np.any(mask):
            return
        self.energy[mask] = 0.0
        self.velocity[mask] = 0.0
        self.phase[mask] = 0.0
        self.omega[mask] = 0.0
        self.genome_fragment[mask] = -1
        self.genome_len[mask] = 0
        self.genome_weight[mask] = 0.0
        self.ifs_fragment[mask] = 0.0
        self.ifs_weight[mask] = 0.0
        self.emitter_id[mask] = -1
