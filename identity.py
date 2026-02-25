from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EmeraConfig


_WTE_CACHE: dict[str, np.ndarray] = {}


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def hadamard_matrix(n: int) -> np.ndarray:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Hadamard size must be a positive power of two.")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]]).astype(np.float32)
    return h


def random_orthonormal_projection(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    m = rng.normal(size=(rows, cols)).astype(np.float32)
    q, _ = np.linalg.qr(m, mode="reduced")
    return q[:, :cols].astype(np.float32)


def _load_gpt2_wte(model_name: str) -> np.ndarray:
    cached = _WTE_CACHE.get(model_name)
    if cached is not None:
        return cached.copy()
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gpt2 token space requires `huggingface_hub` and `safetensors`."
        ) from exc

    files = list_repo_files(model_name)
    safes = [f for f in files if f.endswith(".safetensors")]
    if not safes:
        raise RuntimeError(f"No .safetensors weights found for model '{model_name}'.")
    preferred = ["model.safetensors"]
    chosen = None
    for name in preferred:
        if name in safes:
            chosen = name
            break
    if chosen is None:
        chosen = safes[0]
    path = hf_hub_download(repo_id=model_name, filename=chosen)
    with safe_open(path, framework="np") as sf:
        keys = list(sf.keys())
        for key in ("transformer.wte.weight", "wte.weight", "model.embed_tokens.weight"):
            if key in keys:
                w = np.asarray(sf.get_tensor(key), dtype=np.float32)
                _WTE_CACHE[model_name] = w
                return w.copy()
        # Fallback for custom checkpoints: first tensor with token embedding semantics.
        for key in keys:
            if "wte" in key or "embed_tokens" in key:
                w = np.asarray(sf.get_tensor(key), dtype=np.float32)
                _WTE_CACHE[model_name] = w
                return w.copy()
    raise RuntimeError(f"Could not locate GPT-2 input embedding tensor in {chosen}.")


def _ifs_from_latent(z: np.ndarray, cfg: EmeraConfig) -> np.ndarray:
    out = np.zeros((z.shape[0], cfg.num_ifs, 2, 3), dtype=np.float32)
    d = z.shape[1]
    for tid in range(z.shape[0]):
        vec = z[tid]
        for k in range(cfg.num_ifs):
            i0 = (2 + 3 * k) % d
            i1 = (3 + 3 * k) % d
            i2 = (4 + 3 * k) % d
            angle = np.pi * vec[i0]
            scale = 0.45 + 0.35 * abs(vec[i1])
            c = np.cos(angle)
            s = np.sin(angle)
            a = np.array([[c, -s], [s, c]], dtype=np.float32) * scale
            b = np.array([0.18 * vec[i1], 0.18 * vec[i2]], dtype=np.float32)
            out[tid, k, :, :2] = a
            out[tid, k, :, 2] = b
    return out


@dataclass
class BaseIdentity:
    latent: np.ndarray  # [512, d_latent]
    beacon: np.ndarray  # [512, gap_dim]
    ifs: np.ndarray  # [512, num_ifs, 2, 3]
    attractor0: np.ndarray  # [512, 2]
    phase0: np.ndarray  # [512]
    omega0: np.ndarray  # [512]


def create_base_identity(cfg: EmeraConfig, rng: np.random.Generator) -> BaseIdentity:
    if cfg.token_space == "gpt2":
        wte = _load_gpt2_wte(cfg.gpt2_model_name)
        vocab_size, emb_dim = int(wte.shape[0]), int(wte.shape[1])
        if cfg.base_tokens > vocab_size:
            raise ValueError(
                f"base_tokens={cfg.base_tokens} exceeds GPT-2 vocab size {vocab_size}."
            )
        if cfg.base_tokens < vocab_size:
            wte = wte[: cfg.base_tokens]
        p = random_orthonormal_projection(rng, emb_dim, cfg.d_latent)
        z = _normalize_rows(_normalize_rows(wte) @ p)
    else:
        h = hadamard_matrix(cfg.base_tokens) / np.sqrt(float(cfg.base_tokens))
        p = random_orthonormal_projection(rng, cfg.base_tokens, cfg.d_latent)
        z = _normalize_rows(h @ p)
    beacon = _normalize_rows(z[:, : cfg.gap_dim])
    ifs = _ifs_from_latent(z, cfg)
    attractor0 = z[:, :2].astype(np.float32)
    phase0 = np.arctan2(attractor0[:, 1], attractor0[:, 0]).astype(np.float32)
    omega0 = (cfg.omega_base + cfg.omega_jitter * z[:, 2]).astype(np.float32)
    return BaseIdentity(
        latent=z.astype(np.float32),
        beacon=beacon.astype(np.float32),
        ifs=ifs.astype(np.float32),
        attractor0=attractor0,
        phase0=phase0,
        omega0=omega0,
    )
