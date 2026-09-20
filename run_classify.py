#!/usr/bin/env python3
"""Train and evaluate the Emera classifier substrate on a labelled task.

    python run_classify.py --task kjv-genre --steps 8000

Reports rolling training accuracy while the population evolves, then held-out
accuracy against a majority-class and a hashed n-gram centroid baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from classifier import ClassifierConfig, EmeraClassifier
from config import EmeraConfig
from tasks import centroid_baseline, load_task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=str, default="kjv-genre")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-label-train", type=int, default=1200)
    p.add_argument("--per-label-eval", type=int, default=200)
    p.add_argument("--eval-size", type=int, default=400)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=0)

    p.add_argument("--population", type=int, default=None)
    p.add_argument("--max-population", type=int, default=None)
    p.add_argument("--k-rounds", type=int, default=3)
    p.add_argument("--d-latent", type=int, default=32)
    p.add_argument("--gap-dim", type=int, default=16)
    p.add_argument("--gap-len", type=int, default=128)
    p.add_argument("--reservoir-init", type=float, default=None)
    p.add_argument("--metabolic-tax-rate", type=float, default=None)
    p.add_argument("--rake-frac", type=float, default=None)
    p.add_argument("--wake-feature-weight", type=float, default=None)
    p.add_argument("--context-gain", type=float, default=None)
    p.add_argument("--discovery-spawn-prob", type=float, default=None)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--show-population", type=int, default=25)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    task = load_task(
        args.task,
        data_dir=Path(args.data_dir),
        per_label_train=args.per_label_train,
        per_label_eval=args.per_label_eval,
        seed=args.seed,
    )
    print(task.describe())

    cfg = replace(
        EmeraConfig(),
        seed=args.seed,
        d_latent=args.d_latent,
        gap_dim=args.gap_dim,
        gap_len=args.gap_len,
        k_rounds=args.k_rounds,
    )
    overrides = {
        "initial_population": args.population,
        "max_population": args.max_population,
        "reservoir_init": args.reservoir_init,
        "metabolic_tax_rate": args.metabolic_tax_rate,
        "rake_frac": args.rake_frac,
        "context_gain_init": args.context_gain,
        "discovery_spawn_prob": args.discovery_spawn_prob,
    }
    if args.wake_feature_weight is not None:
        overrides["wake_feature_weight"] = args.wake_feature_weight
        overrides["wake_resonance_weight"] = 1.0 - args.wake_feature_weight
    ccfg = replace(
        ClassifierConfig(),
        seed=args.seed,
        **{k: v for k, v in overrides.items() if v is not None},
    )

    model = EmeraClassifier(task=task, cfg=cfg, ccfg=ccfg)
    rng = np.random.default_rng(args.seed + 1)

    val_pool = task.val[: args.eval_size] if args.eval_size > 0 else task.val
    history: list[dict] = []

    for step in range(1, args.steps + 1):
        ex = task.train[int(rng.integers(0, len(task.train)))]
        report = model.train_step(ex)
        if args.log_every > 0 and step % args.log_every == 0:
            s = model.stats()
            print(
                f"step {step:6d} | roll_acc {s['rolling_accuracy']:.3f} "
                f"| pop {s['population']:4d} | res {s['reservoir']:8.1f} "
                f"| E {s['total_energy']:8.1f} (drift {s['energy_drift']:+.2e}) "
                f"| voters {report.voters:4d}/{report.voters + report.abstained:4d} "
                f"| b/d {s['births']}/{s['deaths']}"
            )
            history.append({"step": step, **s})
        if args.eval_every > 0 and step % args.eval_every == 0:
            val = model.evaluate(val_pool)
            print(f"           val acc {val['accuracy']:.4f} nll {val['mean_nll']:.3f}")

    train_secs = time.time() - t_start

    test = model.evaluate(task.test)
    val = model.evaluate(val_pool)
    print()
    print(f"train time        {train_secs:8.1f}s ({args.steps} steps)")
    print(f"val   accuracy    {val['accuracy']:.4f}")
    print(f"test  accuracy    {test['accuracy']:.4f}  (macro {test['macro_accuracy']:.4f})")
    print(f"test  mean NLL    {test['mean_nll']:.4f}")
    print(f"mean voters/ex    {test['mean_voters']:.1f} of {len(model.population)} organisms")

    baselines: dict[str, float] = {"majority": task.majority_baseline()}
    if not args.no_baselines:
        baselines["ngram_centroid"] = centroid_baseline(task)
    for name, value in baselines.items():
        print(f"baseline {name:<16} {value:.4f}")

    print()
    print("per-class recall:")
    for name, value in zip(task.labels, test["per_class"]):
        print(f"  {name:<12} {value:.3f}")

    if args.show_population > 0:
        print()
        print(f"top {args.show_population} organisms (pattern -> label):")
        for row in model.describe_population(args.show_population):
            print(row)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": task.name,
            "labels": task.labels,
            "steps": args.steps,
            "train_seconds": train_secs,
            "config": asdict(cfg),
            "classifier_config": asdict(ccfg),
            "val": {k: v for k, v in val.items() if k != "confusion"},
            "test": test,
            "baselines": baselines,
            "history": history,
            "population": model.describe_population(args.show_population),
            "stats": model.stats(),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
