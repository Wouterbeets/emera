from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np

from config import EmeraConfig

_JAX_STATE: dict[str, Any] = {
    "checked": False,
    "ok": False,
    "has_accel": False,
    "jax": None,
    "jnp": None,
    "kernel": None,
}


def _normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _ensure_jax_state() -> dict[str, Any]:
    if bool(_JAX_STATE.get("checked", False)):
        return _JAX_STATE
    _JAX_STATE["checked"] = True
    try:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import jax
            import jax.numpy as jnp

            devices = jax.devices()

        has_accel = any(str(getattr(d, "platform", "")).lower() != "cpu" for d in devices)

        @jax.jit
        def _read_many_jax_kernel(
            points: Any,
            phase: Any,
            energy: Any,
            receiver_signature: Any,
            resonance_width: Any,
            receiver_phase: Any,
            phase_coupling: Any,
            valid_mask: Any,
        ) -> tuple[Any, Any]:
            eff_energy = jnp.where(energy > 1e-9, energy, 0.0)
            sum_energy = jnp.sum(eff_energy)
            valid = jnp.clip(valid_mask.reshape(-1), 0.0, 1.0).astype(jnp.float32)

            pts_norm = points / jnp.maximum(jnp.linalg.norm(points, axis=1, keepdims=True), 1e-8)
            pts_norm = jnp.nan_to_num(pts_norm, nan=0.0, posinf=0.0, neginf=0.0)

            recv = receiver_signature / jnp.maximum(
                jnp.linalg.norm(receiver_signature, axis=1, keepdims=True),
                1e-8,
            )
            recv = jnp.nan_to_num(recv, nan=0.0, posinf=0.0, neginf=0.0)
            recv = recv * valid[:, None]

            sim = jnp.clip(recv @ pts_norm.T, -1.0, 1.0)
            width = jnp.maximum(resonance_width[:, None], 1e-3)
            geo = jnp.exp(-((1.0 - sim) ** 2) / (2.0 * width * width))

            phase_align = 0.5 * (1.0 + jnp.cos(phase[None, :] - receiver_phase[:, None]))
            pc = phase_coupling[:, None]
            phase_gate = (1.0 - pc) + pc * phase_align

            w = eff_energy[None, :] * geo * phase_gate
            w = jnp.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w = w * valid[:, None]
            total_w = jnp.sum(w, axis=1)

            resonance = (w @ points) / jnp.maximum(total_w[:, None], 1e-9)
            strength = jnp.clip(total_w / jnp.maximum(sum_energy, 1e-9), 0.0, 1.0)

            has_energy = sum_energy > 1e-9
            resonance = jnp.where(has_energy, resonance, jnp.zeros_like(resonance))
            strength = jnp.where(has_energy, strength, jnp.zeros_like(strength))
            resonance = resonance * valid[:, None]
            strength = strength * valid
            return resonance.astype(jnp.float32), strength.astype(jnp.float32)

        _JAX_STATE["ok"] = True
        _JAX_STATE["has_accel"] = bool(has_accel)
        _JAX_STATE["jax"] = jax
        _JAX_STATE["jnp"] = jnp
        _JAX_STATE["kernel"] = _read_many_jax_kernel
    except Exception:
        _JAX_STATE["ok"] = False
        _JAX_STATE["has_accel"] = False
        _JAX_STATE["jax"] = None
        _JAX_STATE["jnp"] = None
        _JAX_STATE["kernel"] = None
    return _JAX_STATE


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
        # Full substrate capsule payload: organism state lives in gap slots.
        self.capsule_token_id = np.full((cfg.gap_len,), -1, dtype=np.int64)
        self.capsule_parent_a = np.full((cfg.gap_len,), -1, dtype=np.int64)
        self.capsule_parent_b = np.full((cfg.gap_len,), -1, dtype=np.int64)
        self.capsule_birth_step = np.zeros((cfg.gap_len,), dtype=np.int64)
        self.capsule_inactivity_steps = np.zeros((cfg.gap_len,), dtype=np.int32)
        self.capsule_low_energy_steps = np.zeros((cfg.gap_len,), dtype=np.int32)
        self.capsule_lineage_depth = np.zeros((cfg.gap_len,), dtype=np.int16)
        self.capsule_max_paid_bet = np.zeros((cfg.gap_len,), dtype=np.float32)
        self.capsule_max_silent_correct = np.zeros((cfg.gap_len,), dtype=np.int32)
        self.capsule_state_vec = np.zeros((cfg.gap_len, cfg.d_latent), dtype=np.float32)
        self.capsule_signature = np.zeros((cfg.gap_len, cfg.gap_dim), dtype=np.float32)
        self.capsule_proposal_drift = np.zeros((cfg.gap_len, cfg.d_latent), dtype=np.float32)
        self.capsule_traits = np.zeros((cfg.gap_len, 8), dtype=np.float32)
        self.capsule_identity = np.full((cfg.gap_len, max(int(cfg.proposal_lmax), 1)), -1, dtype=np.int32)
        self.capsule_identity_len = np.zeros((cfg.gap_len,), dtype=np.int16)
        self.capsule_ifs = np.zeros((cfg.gap_len, int(cfg.num_ifs), 2, 3), dtype=np.float32)
        self.ptr = 0
        self.read_backend = self._resolve_read_backend(str(cfg.gap_read_backend))

    def _resolve_read_backend(self, requested: str) -> str:
        req = str(requested).strip().lower()
        if req not in {"auto", "numpy", "jax"}:
            req = "auto"
        if req == "numpy":
            return "numpy"
        jax_state = _ensure_jax_state()
        if req == "jax":
            return "jax" if bool(jax_state.get("ok", False)) else "numpy"
        if bool(jax_state.get("ok", False)) and bool(jax_state.get("has_accel", False)):
            est_work = (
                int(self.cfg.gap_len)
                * int(self.cfg.gap_dim)
                * int(max(self.cfg.gap_read_batch_size, 1))
            )
            # Auto mode prefers NumPy for tiny kernels where host/device overhead dominates.
            if est_work >= 500_000:
                return "jax"
        return "numpy"

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
        capsule_token: Any | None = None,
        lineage_depth: int | None = None,
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
        if capsule_token is not None:
            self._store_capsule(p, capsule_token, lineage_depth)
        self.ptr = (p + 1) % self.cfg.gap_len

    def _store_capsule(self, slot: int, token: Any, lineage_depth: int | None = None) -> None:
        s = int(slot)
        ident = np.asarray(getattr(token, "identity_bytes"), dtype=np.int32).reshape(-1)
        n = min(int(ident.size), int(self.capsule_identity.shape[1]))
        self.capsule_token_id[s] = int(getattr(token, "token_id"))
        self.capsule_parent_a[s] = int(getattr(token, "parent_a", -1))
        self.capsule_parent_b[s] = int(getattr(token, "parent_b", -1))
        self.capsule_birth_step[s] = int(getattr(token, "birth_step", 0))
        self.capsule_inactivity_steps[s] = int(getattr(token, "inactivity_steps", 0))
        self.capsule_low_energy_steps[s] = int(getattr(token, "low_energy_steps", 0))
        if lineage_depth is not None:
            self.capsule_lineage_depth[s] = int(lineage_depth)
        self.capsule_max_paid_bet[s] = float(getattr(token, "max_paid_bet", 0.0))
        self.capsule_max_silent_correct[s] = int(getattr(token, "max_silent_correct", 0))
        self.capsule_state_vec[s] = np.asarray(getattr(token, "state_vec"), dtype=np.float32)
        self.capsule_signature[s] = np.asarray(getattr(token, "signature"), dtype=np.float32)
        self.capsule_proposal_drift[s] = np.asarray(getattr(token, "proposal_drift"), dtype=np.float32)
        self.capsule_traits[s, 0] = float(getattr(token, "activation_threshold", 0.5))
        self.capsule_traits[s, 1] = float(getattr(token, "emission_amplitude", 0.4))
        self.capsule_traits[s, 2] = float(getattr(token, "emission_decay", 0.15))
        self.capsule_traits[s, 3] = float(getattr(token, "silence_growth_rate", 1.0))
        self.capsule_traits[s, 4] = float(getattr(token, "resonance_width", 0.45))
        self.capsule_traits[s, 5] = float(getattr(token, "phase_coupling", 0.25))
        self.capsule_traits[s, 6] = float(getattr(token, "velocity_coupling", 0.20))
        self.capsule_traits[s, 7] = float(getattr(token, "proposal_length_bias", 0.0))
        self.capsule_identity[s].fill(-1)
        if n > 0:
            vmax = 255 if self.cfg.token_space == "byte_parity" else max(int(self.cfg.base_tokens) - 1, 0)
            self.capsule_identity[s, :n] = np.clip(ident[:n], 0, vmax).astype(np.int32, copy=False)
        self.capsule_identity_len[s] = int(n)
        self.capsule_ifs[s] = np.asarray(getattr(token, "ifs"), dtype=np.float32)

    def clear_slot(self, slot: int) -> None:
        s = int(slot)
        self.points[s] = 0.0
        self.velocity[s] = 0.0
        self.phase[s] = 0.0
        self.omega[s] = 0.0
        self.energy[s] = 0.0
        self.genome_fragment[s] = -1
        self.genome_len[s] = 0
        self.genome_weight[s] = 0.0
        self.ifs_fragment[s] = 0.0
        self.ifs_weight[s] = 0.0
        self.emitter_id[s] = -1
        self.step_idx[s] = 0
        self.round_idx[s] = 0
        self.capsule_token_id[s] = -1
        self.capsule_parent_a[s] = -1
        self.capsule_parent_b[s] = -1
        self.capsule_birth_step[s] = 0
        self.capsule_inactivity_steps[s] = 0
        self.capsule_low_energy_steps[s] = 0
        self.capsule_lineage_depth[s] = 0
        self.capsule_max_paid_bet[s] = 0.0
        self.capsule_max_silent_correct[s] = 0
        self.capsule_state_vec[s] = 0.0
        self.capsule_signature[s] = 0.0
        self.capsule_proposal_drift[s] = 0.0
        self.capsule_traits[s] = 0.0
        self.capsule_identity[s] = -1
        self.capsule_identity_len[s] = 0
        self.capsule_ifs[s] = 0.0

    def live_capsule_slots(self, min_energy: float = 0.0) -> np.ndarray:
        return np.where(
            (self.capsule_token_id >= 0) & (self.energy > float(min_energy))
        )[0].astype(np.int32, copy=False)

    def _read_many_numpy(
        self,
        receiver_signature: np.ndarray,
        resonance_width: np.ndarray,
        receiver_phase: np.ndarray,
        phase_coupling: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(receiver_signature, dtype=np.float32)
        widths = np.asarray(resonance_width, dtype=np.float32).reshape(-1)
        rphase = np.asarray(receiver_phase, dtype=np.float32).reshape(-1)
        pc = np.asarray(phase_coupling, dtype=np.float32).reshape(-1)

        n = int(sig.shape[0])
        if n <= 0:
            return (
                np.zeros((0, int(self.cfg.gap_dim)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        if valid_mask is None:
            valid = np.ones((n,), dtype=np.float32)
        else:
            valid = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
            if valid.size != n:
                raise ValueError("valid_mask must match receiver batch size")
            valid = np.clip(valid, 0.0, 1.0)

        energy = np.where(self.energy > 1e-9, self.energy, 0.0).astype(np.float32)
        sum_energy = float(np.sum(energy))
        if sum_energy <= 1e-9:
            return (
                np.zeros((n, int(self.cfg.gap_dim)), dtype=np.float32),
                np.zeros((n,), dtype=np.float32),
            )

        pts = self.points.astype(np.float32, copy=False)
        pts_norm = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-8)
        pts_norm = np.nan_to_num(pts_norm, nan=0.0, posinf=0.0, neginf=0.0)

        recv = sig / np.maximum(np.linalg.norm(sig, axis=1, keepdims=True), 1e-8)
        recv = np.nan_to_num(recv, nan=0.0, posinf=0.0, neginf=0.0)
        recv = recv * valid.reshape(-1, 1)

        sim = np.clip(recv @ pts_norm.T, -1.0, 1.0)
        width = np.maximum(widths.reshape(-1, 1), 1e-3)
        geo = np.exp(-((1.0 - sim) ** 2) / (2.0 * width * width)).astype(np.float32)

        phase_align = 0.5 * (1.0 + np.cos(self.phase.reshape(1, -1) - rphase.reshape(-1, 1)))
        phase_gate = (1.0 - pc.reshape(-1, 1)) + pc.reshape(-1, 1) * phase_align.astype(np.float32)

        w = energy.reshape(1, -1) * geo * phase_gate
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        w *= valid.reshape(-1, 1)
        total_w = np.sum(w, axis=1, dtype=np.float64).astype(np.float32)

        resonance = (w @ pts).astype(np.float32)
        resonance /= np.maximum(total_w.reshape(-1, 1), 1e-9)
        resonance *= valid.reshape(-1, 1)

        strength = np.clip(total_w / max(sum_energy, 1e-9), 0.0, 1.0).astype(np.float32)
        strength *= valid
        return resonance, strength

    def _read_many_jax(
        self,
        receiver_signature: np.ndarray,
        resonance_width: np.ndarray,
        receiver_phase: np.ndarray,
        phase_coupling: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        jax_state = _ensure_jax_state()
        if not bool(jax_state.get("ok", False)):
            return self._read_many_numpy(
                receiver_signature=receiver_signature,
                resonance_width=resonance_width,
                receiver_phase=receiver_phase,
                phase_coupling=phase_coupling,
                valid_mask=valid_mask,
            )
        if valid_mask is None:
            valid = np.ones((int(np.asarray(receiver_signature).shape[0]),), dtype=np.float32)
        else:
            valid = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
        jnp = jax_state["jnp"]
        kernel = jax_state["kernel"]
        r, s = kernel(
            jnp.asarray(self.points, dtype=jnp.float32),
            jnp.asarray(self.phase, dtype=jnp.float32),
            jnp.asarray(self.energy, dtype=jnp.float32),
            jnp.asarray(receiver_signature, dtype=jnp.float32),
            jnp.asarray(resonance_width, dtype=jnp.float32),
            jnp.asarray(receiver_phase, dtype=jnp.float32),
            jnp.asarray(phase_coupling, dtype=jnp.float32),
            jnp.asarray(valid, dtype=jnp.float32),
        )
        return np.asarray(r, dtype=np.float32), np.asarray(s, dtype=np.float32)

    def read_many(
        self,
        receiver_signature: np.ndarray,
        resonance_width: np.ndarray,
        receiver_phase: np.ndarray,
        phase_coupling: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sig = np.asarray(receiver_signature, dtype=np.float32)
        if sig.ndim != 2 or sig.shape[1] != int(self.cfg.gap_dim):
            raise ValueError(
                f"receiver_signature must have shape [N, {int(self.cfg.gap_dim)}], got {sig.shape}"
            )
        n = int(sig.shape[0])
        widths = np.asarray(resonance_width, dtype=np.float32).reshape(-1)
        rphase = np.asarray(receiver_phase, dtype=np.float32).reshape(-1)
        pc = np.asarray(phase_coupling, dtype=np.float32).reshape(-1)
        if widths.size != n or rphase.size != n or pc.size != n:
            raise ValueError("batch read inputs must have matching leading dimension")
        valid: np.ndarray | None = None
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
            if valid.size != n:
                raise ValueError("valid_mask must match receiver batch size")
        if self.read_backend == "jax":
            return self._read_many_jax(sig, widths, rphase, pc, valid)
        return self._read_many_numpy(sig, widths, rphase, pc, valid)

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
        include_contributions: bool = True,
    ) -> GapRead:
        active, w = self._weights(
            receiver_signature=receiver_signature,
            resonance_width=resonance_width,
            receiver_phase=receiver_phase,
            phase_coupling=phase_coupling,
        )
        total_w = float(np.sum(w)) if w.size else 0.0
        if w.size == 0 or total_w <= 1e-9:
            return GapRead(
                resonance=np.zeros((self.cfg.gap_dim,), dtype=np.float32),
                strength=0.0,
                contributions={},
            )
        pts = self.points[active]
        resonance = np.sum(pts * w[:, None], axis=0) / max(total_w, 1e-9)

        active_energy = self.energy[active]
        strength = float(total_w / max(np.sum(active_energy), 1e-9))
        strength = float(np.clip(strength, 0.0, 1.0))

        if include_contributions:
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
        for s in np.where(mask)[0].tolist():
            self.clear_slot(int(s))
