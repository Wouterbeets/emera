#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from config import EmeraConfig
from engine import EmeraEngine, StepResult


def _decode_identity(
    engine: EmeraEngine, identity_values: np.ndarray, max_chars: int = 48
) -> str:
    vals = np.asarray(identity_values, dtype=np.int32).tolist()
    text = engine.decode_tokens_text(vals, max_chars=max_chars)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _parse_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def _parse_float(v: str | None, default: float) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _parse_int(
    v: str | None, default: int, lo: int | None = None, hi: int | None = None
) -> int:
    if v is None:
        out = default
    else:
        try:
            out = int(v)
        except ValueError:
            out = default
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out


class Trainer:
    def __init__(self, cfg: EmeraConfig, steps_per_tick: int, sleep_s: float):
        self.engine = EmeraEngine(cfg)
        self.world_bytes: np.ndarray | None = None
        if self.engine.cfg.token_space == "byte_parity":
            self.world_bytes = (self.engine.world.tokens & 0xFF).astype(np.uint8)
        self.lock = threading.RLock()
        self.steps_per_tick = max(int(steps_per_tick), 1)
        self.sleep_s = max(float(sleep_s), 0.0)
        self.running = True
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self.last_result: StepResult | None = None
        self.last_inference: dict[str, Any] | None = None
        self.total_steps = 0
        self.error = ""
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name="emera-trainer"
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def toggle(self) -> bool:
        with self.lock:
            self.running = not self.running
            return self.running

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.running:
                time.sleep(0.05)
                continue
            try:
                for _ in range(self.steps_per_tick):
                    if self.stop_event.is_set():
                        break
                    with self.lock:
                        self.last_result = self.engine.step()
                        self.total_steps += 1
            except Exception as exc:  # pragma: no cover
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.running = False
                time.sleep(0.2)
            if self.sleep_s > 0.0:
                time.sleep(self.sleep_s)
            else:
                time.sleep(0.0)

    def _frontier_line(self) -> tuple[str, str, str]:
        if self.world_bytes is None:
            n = int(self.engine.world.length)
            if n == 0:
                return "", "", ""
            c = int(self.engine.world.cursor) % n
            window = 10
            idx = (c + np.arange(-window, window + 1, dtype=np.int64)) % n
            vals = self.engine.world.tokens[idx].astype(np.int32).tolist()
            left = self.engine.decode_tokens_text(vals[:window], max_chars=200)
            cur = self.engine.decode_tokens_text([vals[window]], max_chars=60)
            right = self.engine.decode_tokens_text(vals[window + 1 :], max_chars=200)
            marked = f"{left}[{cur}]{right}"
            frontier_token = int(self.engine.world.peek(1)[0]) if n > 0 else -1
            frontier_tag = (
                f"{frontier_token}:{cur.encode('unicode_escape').decode('ascii')}"
            )
            return "TOKEN_STREAM [*]", marked, frontier_tag

        n = int(self.world_bytes.size)
        if n == 0:
            return "", "", ""
        c = int(self.engine.world.cursor) % n
        start = c
        while start > 0 and int(self.world_bytes[start - 1]) != 10:
            start -= 1
        end = c
        while end < (n - 1) and int(self.world_bytes[end]) != 10:
            end += 1
        b = self.world_bytes
        pre = b[start:c].tobytes().decode("utf-8", errors="replace")
        cur = bytes([int(b[c])]).decode("utf-8", errors="replace")
        post = b[c + 1 : end].tobytes().decode("utf-8", errors="replace")
        marked = f"{pre}[{cur}]{post}"
        clean = (pre + cur + post).strip()
        m = re.match(r"^\s*([1-3]?\s?[A-Za-z]+)\s+(\d+):(\d+)", clean)
        if m:
            book = m.group(1).upper()
            chapter = m.group(2)
            verse = m.group(3)
            location = f"{book} [{chapter}] :{verse}"
        else:
            location = "UNKNOWN [*]"
        frontier_token = int(self.engine.world.peek(1)[0]) if n > 0 else -1
        frontier_byte = frontier_token & 0xFF
        frontier_hex = f"0x{frontier_byte:02x}"
        return location, marked, frontier_hex

    def status_snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            elapsed = max(now - self.started_at, 1e-9)
            steps_per_s = float(self.total_steps) / elapsed
            stats = self.last_result.stats if self.last_result is not None else {}
            snap = self.engine.metrics.snapshot()
            location, marked, frontier_hex = self._frontier_line()
            return {
                "running": bool(self.running),
                "error": self.error,
                "step": int(self.engine.step_idx),
                "active": int(len(self.engine.super_tokens)),
                "proposers": int(stats.get("proposers", 0)),
                "winner": int(stats.get("winner_id", -1)),
                "match": int(stats.get("match_len", 0)),
                "advance": int(stats.get("advance_len", 0)),
                "return": float(stats.get("realized_return", 0.0)),
                "jackpot": float(stats.get("jackpot", 0.0)),
                "reservoir": float(stats.get("reservoir", 0.0)),
                "drift": float(stats.get("energy_drift", 0.0)),
                "steps_per_s": steps_per_s,
                "location": location,
                "frontier_line": marked,
                "frontier_hex": frontier_hex,
                "glide": float(snap.get("glide_ratio", 0.0)),
                "epa": float(snap.get("energy_per_advance", 0.0)),
                "root_frac": float(snap.get("root_only_fraction", 1.0)),
                "caps_nonroot": int(snap.get("nonroot_live_capsules", 0)),
                "caps_half_life": float(snap.get("capsule_half_life", 0.0)),
                "caps_t1k": float(snap.get("lineage_persistence_1k", 0.0)),
                "chaos_avg_sub": float(snap.get("chaos_avg_substeps", 0.0)),
                "chaos_energy": float(snap.get("chaos_energy_spent", 0.0)),
            }

    def supertokens(
        self, sort_by: str, desc: bool, min_energy: float, query: str, limit: int
    ) -> list[dict[str, Any]]:
        with self.lock:
            step_idx = int(self.engine.step_idx)
            q = query.lower().strip()
            rows: list[dict[str, Any]] = []
            for token in self.engine.super_tokens.values():
                energy = float(token.energy)
                if energy < min_energy:
                    continue
                text = _decode_identity(
                    self.engine,
                    token.identity_bytes,
                    max_chars=64,
                )
                if q and q not in text.lower() and q not in str(token.token_id):
                    continue
                age = max(step_idx - int(token.birth_step), 0)
                rs = float(self.engine.last_resonance_strength.get(token.token_id, 0.0))
                _, conf, _ = self.engine._raw_decode_for_token(token, rs)
                rows.append(
                    {
                        "id": int(token.token_id),
                        "age": int(age),
                        "energy": float(energy),
                        "inactive": int(token.inactivity_steps),
                        "len": int(
                            np.asarray(token.identity_bytes, dtype=np.int32).size
                        ),
                        "conf": float(conf),
                        "text": text,
                        "contains_frontier": bool(
                            self.engine._identity_contains_frontier_byte(
                                token.identity_bytes
                            )
                        ),
                        "lineage": "root"
                        if int(token.parent_a) < 0 and int(token.parent_b) < 0
                        else f"{int(token.parent_a)},{int(token.parent_b)}",
                    }
                )

            key_map = {
                "id": lambda r: r["id"],
                "age": lambda r: r["age"],
                "energy": lambda r: r["energy"],
                "inactive": lambda r: r["inactive"],
                "len": lambda r: r["len"],
                "conf": lambda r: r["conf"],
            }
            key_fn = key_map.get(sort_by, key_map["energy"])
            rows.sort(key=key_fn, reverse=desc)
            return rows[: max(limit, 1)]

    def chaos_svg(
        self,
        token_id: int,
        steps: int | None = None,
        width: int = 440,
        height: int = 230,
    ) -> str:
        with self.lock:
            token = self.engine.super_tokens.get(int(token_id))
            if token is None:
                return "<div class='card'><strong>Chaos</strong><p>token not found.</p></div>"
            ifs = np.asarray(token.ifs, dtype=np.float32)
            p = np.asarray(token.state_vec[:2], dtype=np.float32).copy()
            ident = _decode_identity(
                self.engine,
                token.identity_bytes,
                max_chars=64,
            )
            tid = int(token.token_id)
            energy = float(token.energy)
            cfg = self.engine.cfg
            default_steps = int(getattr(cfg, "chaos_svg_default_steps", 2000))
            max_steps = int(getattr(cfg, "chaos_svg_max_steps", max(default_steps, 1)))
            s = default_steps if steps is None else int(steps)
            s = int(np.clip(s, 1, max_steps))

        rng = np.random.default_rng(tid + 1337)
        pts = np.zeros((s, 2), dtype=np.float32)
        for i in range(s):
            k = int(rng.integers(0, ifs.shape[0]))
            a = ifs[k, :, :2]
            b = ifs[k, :, 2]
            p = np.tanh(a @ p + b).astype(np.float32)
            pts[i] = p

        pad = 10.0
        xs = ((pts[:, 0] + 1.0) * 0.5) * (width - 2 * pad) + pad
        ys = (1.0 - (pts[:, 1] + 1.0) * 0.5) * (height - 2 * pad) + pad
        point_str = " ".join(
            f"{x:.2f},{y:.2f}" for x, y in zip(xs.tolist(), ys.tolist())
        )
        return (
            "<div class='card'>"
            f"<strong>Chaos Game | token {tid}</strong>"
            f"<div class='muted'>energy={energy:.3f} steps={s} identity='{html.escape(ident)}'</div>"
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' class='plot'>"
            "<rect x='0' y='0' width='100%' height='100%' fill='#0b0f17'/>"
            f"<polyline points='{point_str}' fill='none' stroke='#5dd6ff' stroke-opacity='0.52' stroke-width='1.0'/>"
            "</svg>"
            "</div>"
        )

    def gap_snapshot(self, limit: int = 32, scatter_limit: int = 220) -> dict[str, Any]:
        with self.lock:
            energy = np.asarray(self.engine.gap.energy, dtype=np.float32).copy()
            pts = np.asarray(self.engine.gap.points, dtype=np.float32).copy()
            emit = np.asarray(self.engine.gap.emitter_id, dtype=np.int64).copy()
            step_idx = np.asarray(self.engine.gap.step_idx, dtype=np.int64).copy()
            round_idx = np.asarray(self.engine.gap.round_idx, dtype=np.int32).copy()
            ptr = int(self.engine.gap.ptr)

        active = np.where(energy > 1e-7)[0]
        if active.size == 0:
            return {"rows": [], "scatter": [], "ptr": ptr}
        order = active[np.argsort(energy[active])[::-1]]
        top = order[: max(limit, 1)]
        rows = [
            {
                "slot": int(i),
                "energy": float(energy[i]),
                "emitter": int(emit[i]),
                "step": int(step_idx[i]),
                "round": int(round_idx[i]),
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
            }
            for i in top.tolist()
        ]
        scatter_idx = order[: max(scatter_limit, 1)]
        max_e = float(np.max(energy[scatter_idx])) if scatter_idx.size else 1.0
        scatter = []
        for i in scatter_idx.tolist():
            e = float(energy[i])
            ratio = float(np.clip(e / max(max_e, 1e-9), 0.0, 1.0))
            scatter.append(
                {
                    "x": float(pts[i, 0]),
                    "y": float(pts[i, 1]),
                    "e": e,
                    "r": 1.0 + 3.0 * np.sqrt(ratio),
                    "h": int(200 - 150 * ratio),
                }
            )
        return {"rows": rows, "scatter": scatter, "ptr": ptr}

    def run_inference(
        self,
        prompt: str,
        new_tokens: int,
        passes: int,
        window: int,
        top_k: int,
        temperature: float,
    ) -> dict[str, Any]:
        with self.lock:
            out = self.engine.infer_generate(
                prompt=prompt,
                max_new_tokens=int(new_tokens),
                diffusion_passes=int(passes),
                window_size=int(window),
                top_k=int(top_k),
                temperature=float(temperature),
            )
            self.last_inference = out
            return out


class DashboardHandler(BaseHTTPRequestHandler):
    trainer: Trainer | None = None

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        return

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = parsed.path
        if path == "/":
            self._send_html(self._render_index())
            return
        if path == "/partials/status":
            self._send_html(self._render_status())
            return
        if path == "/partials/supertokens":
            self._send_html(self._render_supertokens(qs))
            return
        if path == "/partials/chaos":
            self._send_html(self._render_chaos(qs))
            return
        if path == "/partials/gap":
            self._send_html(self._render_gap(qs))
            return
        if path == "/api/status":
            t = self._trainer()
            status = t.status_snapshot()
            snap = t.engine.metrics.snapshot()
            payload = {
                "step": int(status.get("step", 0)),
                "active_super": int(status.get("active", 0)),
                "energy_per_advance": float(snap.get("energy_per_advance", 0.0)),
                "glide_ratio": float(snap.get("glide_ratio", 0.0)),
                "gap_compression_ratio": float(snap.get("gap_compression_ratio", 1.0)),
                "best_gap_compression_ratio": float(
                    snap.get("best_gap_compression_ratio", 1.0)
                ),
                "max_symbio_depth": int(snap.get("max_symbio_depth", 0)),
                "root_only_fraction": float(snap.get("root_only_fraction", 1.0)),
                "nonroot_live_capsules": int(snap.get("nonroot_live_capsules", 0)),
                "capsule_half_life": float(snap.get("capsule_half_life", 0.0)),
                "lineage_persistence_1k": float(
                    snap.get("lineage_persistence_1k", 0.0)
                ),
                "chaos_energy_spent": float(snap.get("chaos_energy_spent", 0.0)),
                "chaos_avg_substeps": float(snap.get("chaos_avg_substeps", 0.0)),
            }
            self._send_json(payload)
            return
        self._send_html("<h1>404</h1>", status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/action/toggle":
            trainer = self._trainer()
            trainer.toggle()
            self._send_html(self._render_status())
            return
        if parsed.path == "/action/infer":
            form = self._form_data()
            prompt = str(form.get("prompt", "") or "")
            new_tokens = _parse_int(form.get("new_tokens"), default=96, lo=1, hi=2000)
            passes = _parse_int(form.get("passes"), default=2, lo=1, hi=12)
            window = _parse_int(form.get("window"), default=16, lo=1, hi=256)
            top_k = _parse_int(form.get("top_k"), default=16, lo=1, hi=256)
            temperature = _parse_float(form.get("temperature"), default=0.0)
            out = self._trainer().run_inference(
                prompt=prompt,
                new_tokens=new_tokens,
                passes=passes,
                window=window,
                top_k=top_k,
                temperature=temperature,
            )
            self._send_html(self._render_infer_result(out))
            return
        self._send_html("<h1>404</h1>", status=404)

    def _trainer(self) -> Trainer:
        trainer = self.__class__.trainer
        if trainer is None:
            raise RuntimeError("trainer not initialized")
        return trainer

    def _form_data(self) -> dict[str, str]:
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            n = 0
        raw = (
            self.rfile.read(max(n, 0)).decode("utf-8", errors="ignore") if n > 0 else ""
        )
        parsed = parse_qs(raw)
        out: dict[str, str] = {}
        for k, vals in parsed.items():
            out[k] = vals[0] if vals else ""
        return out

    def _render_index(self) -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emera Live Dashboard</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>
    :root { --bg:#0b0f17; --card:#121826; --muted:#7e8aa6; --text:#ecf2ff; --line:#26324a; --good:#61d9a0; --warn:#ffcc66; --bad:#ff6a7a; --accent:#5dd6ff; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, SF Pro Text, Segoe UI, sans-serif; background:linear-gradient(180deg,#0b0f17 0%,#090d15 100%); color:var(--text); }
    .wrap { max-width: 1400px; margin: 20px auto; padding: 0 14px 20px; }
    .grid { display:grid; gap:12px; grid-template-columns: 1.3fr 1fr; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 12px; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .muted { color:var(--muted); font-size: 12px; }
    .k { color: var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .v { font-weight: 700; margin-right: 8px; }
    .btn { border:1px solid var(--line); border-radius:8px; background:#172037; color:var(--text); padding:6px 10px; cursor:pointer; }
    input, select { border:1px solid var(--line); border-radius:8px; background:#0f1422; color:var(--text); padding:6px 8px; }
    table { width:100%; border-collapse: collapse; font-size:12px; }
    th, td { border-bottom:1px solid var(--line); padding:6px; text-align:left; vertical-align:top; }
    th { color:var(--muted); position:sticky; top:0; background:var(--card); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .plot { border:1px solid var(--line); border-radius:8px; width:100%; max-width:100%; height:auto; }
    .pill { font-size:11px; border-radius:999px; padding:2px 8px; border:1px solid var(--line); }
    .ok { color:var(--good); border-color:#295940; }
    .warn { color:var(--warn); border-color:#695526; }
    .bad { color:var(--bad); border-color:#6a2b36; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h2 style="margin: 0 0 12px;">Emera Live Inspection</h2>
    <div id="status" hx-get="/partials/status" hx-trigger="load, every 1s"></div>
    <div class="grid">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <strong>SuperTokens</strong>
          <span class="muted">filter + sort + chaos-game render</span>
        </div>
        <form id="st-filter" class="row" hx-get="/partials/supertokens" hx-target="#supertokens" hx-trigger="change, keyup delay:300ms from:input">
          <label class="muted">Sort <select name="sort"><option value="energy">energy</option><option value="age">age</option><option value="conf">conf</option><option value="len">len</option><option value="inactive">inactive</option><option value="id">id</option></select></label>
          <label class="muted">Desc <input type="checkbox" name="desc" checked /></label>
          <label class="muted">Min energy <input type="number" name="min_energy" value="0" step="0.1" style="width:88px;" /></label>
          <label class="muted">Limit <input type="number" name="limit" value="120" step="1" min="1" max="1000" style="width:78px;" /></label>
          <label class="muted">Chaos steps <input type="number" name="chaos_steps" value="2000" step="100" min="1" max="50000" style="width:96px;" /></label>
          <label class="muted">Query <input type="text" name="q" placeholder="id or text" /></label>
        </form>
        <div id="supertokens" hx-get="/partials/supertokens" hx-include="#st-filter" hx-trigger="load, every 2s"></div>
      </div>
      <div style="display:grid; gap:12px;">
        <div class="card">
          <div class="row" style="justify-content:space-between;">
            <strong>Inference</strong>
            <span class="muted">sliding-window diffusion refinement</span>
          </div>
          <form hx-post="/action/infer" hx-target="#infer-result" hx-swap="innerHTML">
            <label class="muted">Prompt</label>
            <textarea name="prompt" rows="6" style="width:100%; margin:6px 0 8px; border:1px solid var(--line); border-radius:8px; background:#0f1422; color:var(--text); padding:8px;">In the beginning</textarea>
            <div class="row">
              <label class="muted">New <input type="number" name="new_tokens" value="96" min="1" max="2000" style="width:90px;" /></label>
              <label class="muted">Passes <input type="number" name="passes" value="2" min="1" max="12" style="width:80px;" /></label>
              <label class="muted">Window <input type="number" name="window" value="16" min="1" max="256" style="width:80px;" /></label>
              <label class="muted">Top-k <input type="number" name="top_k" value="16" min="1" max="256" style="width:80px;" /></label>
              <label class="muted">Temp <input type="number" name="temperature" value="0.35" step="0.05" min="0" max="2" style="width:80px;" /></label>
              <button class="btn" type="submit">Run</button>
            </div>
          </form>
          <div id="infer-result" class="mono muted" style="margin-top:8px;">run inference to see generated output.</div>
        </div>
        <div id="gap" hx-get="/partials/gap" hx-trigger="load, every 2s"></div>
        <div id="chaos" class="card"><strong>Chaos Game</strong><p class="muted">Click any token's Chaos button to render.</p></div>
      </div>
    </div>
  </div>
</body>
</html>"""

    def _render_status(self) -> str:
        t = self._trainer().status_snapshot()
        state_cls = "ok" if t["running"] else ("bad" if t["error"] else "warn")
        state_txt = "running" if t["running"] else ("error" if t["error"] else "paused")
        err = (
            f"<div class='pill bad mono'>{html.escape(str(t['error']))}</div>"
            if t["error"]
            else ""
        )
        return (
            "<div id='status' class='card'>"
            "<div class='row' style='justify-content:space-between;'>"
            "<div class='row'>"
            f"<span class='pill {state_cls}'>{state_txt}</span>"
            f"<button class='btn' hx-post='/action/toggle' hx-target='#status' hx-swap='outerHTML'>{'Pause' if t['running'] else 'Resume'}</button>"
            "</div>"
            f"<div class='muted'>steps/s {t['steps_per_s']:.1f}</div>"
            "</div>"
            "<div class='row'>"
            f"<span class='k'>Location</span><span class='v mono'>{html.escape(str(t['location']))}</span>"
            f"<span class='k'>Frontier</span><span class='v mono'>{html.escape(str(t['frontier_hex']))}</span>"
            "</div>"
            f"<div class='mono' style='font-size:13px; margin:6px 0 8px;'>{html.escape(str(t['frontier_line']))}</div>"
            "<div class='row'>"
            f"<span class='k'>step</span><span class='v'>{t['step']}</span>"
            f"<span class='k'>active</span><span class='v'>{t['active']}</span>"
            f"<span class='k'>proposers</span><span class='v'>{t['proposers']}</span>"
            f"<span class='k'>winner</span><span class='v'>{t['winner']}</span>"
            f"<span class='k'>match/adv</span><span class='v'>{t['match']}/{t['advance']}</span>"
            f"<span class='k'>R</span><span class='v'>{t['return']:+.3f}</span>"
            f"<span class='k'>payout</span><span class='v'>{t['jackpot']:.3f}</span>"
            f"<span class='k'>reservoir</span><span class='v'>{t['reservoir']:.2f}</span>"
            f"<span class='k'>drift</span><span class='v'>{t['drift']:+.4f}</span>"
            f"<span class='k'>EPA</span><span class='v'>{t['epa']:.4f}</span>"
            f"<span class='k'>glide</span><span class='v'>{t['glide']:.3f}</span>"
            f"<span class='k'>root_frac</span><span class='v'>{t['root_frac']:.3f}</span>"
            f"<span class='k'>caps_nonroot</span><span class='v'>{t['caps_nonroot']}</span>"
            f"<span class='k'>caps_t1k</span><span class='v'>{t['caps_t1k']:.3f}</span>"
            f"<span class='k'>caps_half_life</span><span class='v'>{t['caps_half_life']:.1f}</span>"
            f"<span class='k'>chaos_sub</span><span class='v'>{t['chaos_avg_sub']:.1f}</span>"
            f"<span class='k'>chaos_E</span><span class='v'>{t['chaos_energy']:.4f}</span>"
            "</div>"
            f"{err}"
            "</div>"
        )

    def _render_supertokens(self, qs: dict[str, list[str]]) -> str:
        sort = (qs.get("sort", ["energy"])[0] or "energy").strip()
        desc = _parse_bool(
            qs.get("desc", ["on"])[0] if "desc" in qs else None, default=True
        )
        min_energy = _parse_float(qs.get("min_energy", ["0"])[0], default=0.0)
        limit = _parse_int(qs.get("limit", ["120"])[0], default=120, lo=1, hi=2000)
        chaos_steps = _parse_int(
            qs.get("chaos_steps", ["2000"])[0], default=2000, lo=1, hi=50000
        )
        q = qs.get("q", [""])[0]
        rows = self._trainer().supertokens(
            sort_by=sort, desc=desc, min_energy=min_energy, query=q, limit=limit
        )

        head = (
            "<div class='muted' style='margin:8px 0;'>"
            f"rows={len(rows)} sort={html.escape(sort)} {'desc' if desc else 'asc'} min_energy={min_energy:.3f}"
            "</div>"
        )
        table_head = (
            "<table><thead><tr>"
            "<th>id</th><th>age</th><th>energy</th><th>inactive</th><th>len</th><th>conf</th><th>frontier</th><th>txt</th><th>lineage</th><th></th>"
            "</tr></thead><tbody>"
        )
        body = []
        for r in rows:
            frontier = "yes" if r["contains_frontier"] else "no"
            body.append(
                "<tr>"
                f"<td class='mono'>{r['id']}</td>"
                f"<td>{r['age']}</td>"
                f"<td>{r['energy']:.3f}</td>"
                f"<td>{r['inactive']}</td>"
                f"<td>{r['len']}</td>"
                f"<td>{r['conf']:.3f}</td>"
                f"<td>{frontier}</td>"
                f"<td class='mono'>{html.escape(r['text'])}</td>"
                f"<td class='mono'>{html.escape(r['lineage'])}</td>"
                f"<td><button class='btn' hx-get='/partials/chaos?token_id={r['id']}&steps={chaos_steps}' hx-target='#chaos'>Chaos</button></td>"
                "</tr>"
            )
        tail = "</tbody></table>"
        return head + table_head + "".join(body) + tail

    def _render_chaos(self, qs: dict[str, list[str]]) -> str:
        token_id = _parse_int(qs.get("token_id", ["-1"])[0], default=-1)
        steps = _parse_int(qs.get("steps", ["2000"])[0], default=2000, lo=1, hi=50000)
        if token_id < 0:
            return "<div class='card'><strong>Chaos Game</strong><p class='muted'>missing token_id.</p></div>"
        return self._trainer().chaos_svg(token_id=token_id, steps=steps)

    def _render_gap(self, qs: dict[str, list[str]]) -> str:
        limit = _parse_int(qs.get("limit", ["24"])[0], default=24, lo=1, hi=200)
        scatter_limit = _parse_int(
            qs.get("scatter", ["220"])[0], default=220, lo=16, hi=1000
        )
        snap = self._trainer().gap_snapshot(limit=limit, scatter_limit=scatter_limit)
        rows = snap["rows"]
        scatter = snap["scatter"]
        ptr = int(snap["ptr"])

        width, height = 420, 220
        pad = 10.0
        dots = []
        for p in scatter:
            x = ((p["x"] + 1.0) * 0.5) * (width - 2 * pad) + pad
            y = (1.0 - (p["y"] + 1.0) * 0.5) * (height - 2 * pad) + pad
            dots.append(
                f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{p['r']:.2f}' fill='hsl({p['h']},85%,62%)' fill-opacity='0.65'/>"
            )
        head = (
            "<div class='card'>"
            "<div class='row' style='justify-content:space-between;'>"
            "<strong>Gap Buffer</strong>"
            f"<span class='muted'>ptr={ptr} active={len(scatter)}</span>"
            "</div>"
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' class='plot'>"
            "<rect x='0' y='0' width='100%' height='100%' fill='#0b0f17'/>"
            + "".join(dots)
            + "</svg>"
            "<table><thead><tr><th>slot</th><th>energy</th><th>emitter</th><th>step</th><th>round</th><th>x</th><th>y</th></tr></thead><tbody>"
        )
        body = []
        for r in rows:
            body.append(
                "<tr>"
                f"<td class='mono'>{r['slot']}</td>"
                f"<td>{r['energy']:.4f}</td>"
                f"<td class='mono'>{r['emitter']}</td>"
                f"<td>{r['step']}</td>"
                f"<td>{r['round']}</td>"
                f"<td>{r['x']:.3f}</td>"
                f"<td>{r['y']:.3f}</td>"
                "</tr>"
            )
        return head + "".join(body) + "</tbody></table></div>"

    def _render_infer_result(self, out: dict[str, Any]) -> str:
        text = str(out.get("output_text", "") or "")
        debug = out.get("debug", [])
        debug_html = ""
        if isinstance(debug, list) and debug:
            debug_html = (
                "<div class='muted' style='margin-top:6px;'>"
                + "<br/>".join(html.escape(str(x)) for x in debug[:8])
                + "</div>"
            )
        return (
            "<div class='card'>"
            f"<div class='muted'>mode={html.escape(str(out.get('token_space', '?')))} "
            f"gen={int(out.get('generated_tokens', 0))} "
            f"passes={int(out.get('passes', 1))} "
            f"window={int(out.get('window', 1))} "
            f"top_k={int(out.get('top_k', 1))} "
            f"temp={float(out.get('temperature', 0.0)):.2f}</div>"
            f"<div style='margin-top:8px; white-space:pre-wrap;'>{html.escape(text)}</div>"
            f"{debug_html}"
            "</div>"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emera HTMX dashboard with threaded training"
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--token-space",
        type=str,
        choices=["byte_parity", "gpt2"],
        default="gpt2",
    )
    p.add_argument("--gpt2-model-name", type=str, default="gpt2")
    p.add_argument("--base-tokens", type=int, default=None)
    p.add_argument("--d-latent", type=int, default=32)
    p.add_argument("--gap-dim", type=int, default=16)
    p.add_argument("--gap-len", type=int, default=128)
    p.add_argument("--k-rounds", type=int, default=6)
    p.add_argument("--chaos-substeps-per-round", type=int, default=1)
    p.add_argument("--chaos-svg-default-steps", type=int, default=2000)
    p.add_argument("--chaos-svg-max-steps", type=int, default=50000)
    p.add_argument(
        "--gap-read-backend", type=str, choices=["auto", "numpy", "jax"], default="auto"
    )
    p.add_argument("--gap-read-batch-size", type=int, default=128)
    p.add_argument("--capsule-frontier-window", type=int, default=48)
    p.add_argument("--capsule-mint-parent-pool", type=int, default=10)
    p.add_argument("--season-topology-jitter", type=float, default=0.02)
    p.add_argument("--chaos-min-substeps", type=int, default=1)
    p.add_argument("--chaos-max-substeps", type=int, default=64)
    p.add_argument("--chaos-substep-cost", type=float, default=0.004)
    p.add_argument("--corpus-file", type=str, default="data/bible.txt")
    p.add_argument("--world-len", type=int, default=200_000)
    p.add_argument("--world-vocab-size", type=int, default=50257)
    p.add_argument("--steps-per-tick", type=int, default=1)
    p.add_argument("--sleep-ms", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus = args.corpus_file
    if corpus and not Path(corpus).exists():
        raise FileNotFoundError(f"corpus file not found: {corpus}")

    base_tokens = int(args.base_tokens) if args.base_tokens is not None else None
    world_vocab = int(args.world_vocab_size)
    if args.token_space == "gpt2":
        if base_tokens is None:
            base_tokens = 50257
        if args.world_vocab_size == 512:
            world_vocab = base_tokens

    cfg = replace(
        EmeraConfig(),
        seed=int(args.seed),
        token_space=str(args.token_space),
        gpt2_model_name=str(args.gpt2_model_name),
        base_tokens=int(base_tokens)
        if base_tokens is not None
        else EmeraConfig().base_tokens,
        d_latent=max(int(args.d_latent), 4),
        gap_dim=max(int(args.gap_dim), 4),
        gap_len=max(int(args.gap_len), 4),
        k_rounds=max(int(args.k_rounds), 1),
        chaos_substeps_per_round=max(int(args.chaos_substeps_per_round), 1),
        chaos_svg_default_steps=max(int(args.chaos_svg_default_steps), 1),
        chaos_svg_max_steps=max(
            int(args.chaos_svg_max_steps), int(args.chaos_svg_default_steps), 1
        ),
        gap_read_backend=str(args.gap_read_backend),
        gap_read_batch_size=max(int(args.gap_read_batch_size), 1),
        capsule_frontier_window=max(int(args.capsule_frontier_window), 1),
        capsule_mint_parent_pool=max(int(args.capsule_mint_parent_pool), 2),
        season_topology_jitter=max(float(args.season_topology_jitter), 0.0),
        chaos_min_substeps=max(int(args.chaos_min_substeps), 1),
        chaos_max_substeps=max(int(args.chaos_max_substeps), 1),
        chaos_substep_cost=max(float(args.chaos_substep_cost), 0.0),
        corpus_file=corpus,
        world_len=int(args.world_len),
        world_vocab_size=int(world_vocab),
    )
    cfg.validate()

    trainer = Trainer(
        cfg=cfg,
        steps_per_tick=int(args.steps_per_tick),
        sleep_s=float(args.sleep_ms) / 1000.0,
    )
    trainer.start()

    DashboardHandler.trainer = trainer
    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardHandler)
    print(f"Emera dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        trainer.stop()


if __name__ == "__main__":
    main()
