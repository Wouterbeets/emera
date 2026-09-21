# Emera classifier: measured results

All numbers from `run_classify.py` on this repo's KJV text, 4 seeds (42-45),
8,000 training steps each, evaluated on the full held-out test split
(1,200 verses for genre, 400 for testament). `±` is the standard deviation
across seeds. Reproduce with:

```bash
make classify CLS_TASK=kjv-genre CLS_STEPS=8000 SEED=42
```

## Headline

### kjv-genre — 6 balanced classes, 1,200 test verses

| | accuracy |
|---|---:|
| majority-class baseline | 0.1667 |
| **Emera classifier** | **0.3848 ± 0.0077** |
| hashed 4-gram nearest-centroid baseline | 0.6650 |

Mean NLL 1.5732 ± 0.0071. Population 1,391 ± 52 organisms after 1,093 births
and 214 deaths. 31.5 ± 1.0 of them vote on any given example. 104 s to train.

### kjv-testament — 2 classes, 400 test verses

| | accuracy |
|---|---:|
| majority-class baseline | 0.5000 |
| **Emera classifier** | **0.7112 ± 0.0279** |
| hashed 4-gram nearest-centroid baseline | 0.8200 |

Mean NLL 0.5701 ± 0.0220. Population 596 ± 23, 15.2 ± 0.9 voting per example.
69 s to train.

So: comfortably above chance, comfortably below a linear bag-of-n-grams that
fits in under a second. That is the honest summary.

## What the population looks like

This is the part a dense encoder cannot give you. Richest organisms after
8,000 steps, seed 42 (`pattern → label`, with lifetime vote count and hit
rate):

**kjv-testament**

```
'LORD'   -> old        E=3.71  votes=695  win%=100.0
'LORD'   -> old        E=3.43  votes=951  win%=100.0
'Jesu'   -> new        E=3.34  votes=454  win%=100.0
'LORD'   -> old        E=3.17  votes=841  win%=100.0
'shoul'  -> new        E=2.43  votes=298  win%= 65.4
'came'   -> new        E=2.26  votes=473  win%= 54.3
'And,'   -> new        E=2.00  votes= 46  win%= 71.7
```

**kjv-genre**

```
'childre' -> law       E=3.57  votes=378  win%=38.1
'sons'    -> history   E=3.42  votes=213  win%=50.2
'him,'    -> gospels   E=3.08  votes=276  win%=46.4
'word'    -> prophets  E=3.00  votes=413  win%=31.2
'[was'    -> history   E=2.87  votes=150  win%=51.3
```

Nothing told it that `LORD` is an Old Testament marker or that `Jesu` is a New
Testament one. Those organisms were minted from examples nobody got right,
kept alive by winning bets, and copied because they could afford it.

## Three things worth knowing

### 1. The energy ledger works as a credit-assignment mechanism

Taking the *same* votes from the *same* population and aggregating them three
different ways:

| aggregation | kjv-genre | kjv-testament |
|---|---:|---:|
| plain majority of votes | 0.3031 ± 0.0101 | 0.5925 ± 0.0361 |
| weighted by smoothed hit rate | 0.3808 ± 0.0082 | 0.6850 ± 0.0343 |
| **weighted by energy × return on stake** | **0.3848 ± 0.0077** | **0.7112 ± 0.0279** |

Weighting matters a great deal — 6 to 12 points over counting votes. And the
energy economy matches or slightly beats explicitly tracking how often each
rule has been right, while never recording that statistic: it falls out of who
can pay for what. That is the design's central claim, and on this evidence it
holds.

The ledger is exact. Measured total-energy drift over 8,000 steps is < 1e-7
on a total of 9,000, i.e. float64 rounding.

### 2. The gap field is not earning its place — on this task

Varying how much of the wake decision comes from gap resonance rather than the
literal pattern match, single seed, 8,000 steps, kjv-genre:

| resonance weight in wake | accuracy |
|---|---:|
| 0.00 (pattern only) | 0.3942 |
| 0.20 (default) | 0.3950 |
| 0.50 | 0.3975 |

The seed-to-seed standard deviation is 0.0077, so all three are the same
number. Caveat: this ablates resonance only from the *wake* decision.
Resonance still enters every run through `chaos_step`, which perturbs the state
vector the vote is read from — so this shows the medium does not help decide
*who* speaks, not that it contributes nothing at all. A clean test needs the
chaos path cut too.

### 3. More training does not help; it plateaus

| steps | accuracy (3-4 seeds) | population | voters/example |
|---|---:|---:|---:|
| 8,000 | 0.3890 ± 0.0150 | 1,463 | 33.3 |
| 25,000 | 0.4013 ± 0.0283 | 2,534 | 44.9 |

Tripling the training triples the population and adds a third more voters per
example, for about one point of accuracy against a two-to-three point spread
between seeds. Whatever the limit is, it is not how long it runs.

## Where the limit actually is

Not coverage: 99.96% of test examples get at least one vote.

Not the rules themselves, as far as this can tell: the true label is among
those voted for on 97.3% of examples. That "oracle" figure is weak evidence
though — with 31 voters spread over 6 labels, near-random voting would also
cover the truth most of the time. It rules out a population that has no idea,
not much more.

The likeliest explanation is how thin the evidence per decision is. A verse
has roughly 200 character 4-grams. The centroid baseline uses all of them,
idf-weighted. The population uses about 31 single-pattern votes, each worth one
weighted ballot. It is not that the rules are bad — it is that most of what the
example says is never read by anybody.

That points at conjunctions. A rule that fires on `shall` *and* `LORD`
together is the kind of thing a bag of n-grams structurally cannot represent,
and it is exactly what Emera's symbiogenesis is for. Pair fusion is
implemented in the book-world engine and is *not* implemented here. It is the
obvious next experiment, and the one that would decide whether this
architecture has a real advantage or is a more expensive way to be worse than
a linear model.

## Runtime

Single-core numpy on a 2.8 GHz Xeon, measured at ~870 organisms: **11.6 ms**
per training step, **12.9 ms** per inference. (Inference is not cheaper than
training because it snapshots and restores the population state so that
evaluation cannot perturb it.) Cost scales with the organisms an example
concerns, not with the population, so the 2,500-organism runs are not
meaningfully slower per step.

For scale, not as a like-for-like comparison: laya-mlx reports 13.4 ms P50 for
a typed decision from a 421M-parameter encoder on an M3 Max GPU. Different
hardware, different task, vastly different capability — the point is only that
an evolved rule population is in the same latency neighbourhood while being
about four orders of magnitude smaller and fully readable.
