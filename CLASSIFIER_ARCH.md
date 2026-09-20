# A classifier architecture for Emera

Comparison with [laya-mlx](https://github.com/mizorewww/laya-mlx), and a
working prototype that keeps Emera's substrate but swaps the game.

Status: prototype, measured. See "Results" for what it does and does not do.

---

## 1. What laya-mlx actually is

Laya is a **typed-decision model**: it answers constrained questions in one
bidirectional forward pass, with no autoregressive decoding.

```
state + typed question → bidirectional encoder → decision heads → probabilities
```

The mechanism (`laya_mlx/model.py:202-234`) is worth spelling out, because it is
the part that transfers:

1. The state and the question — including its option list — are rendered into
   one token sequence. Each option leaves a **marker position** in that
   sequence.
2. A ModernBERT/mmBERT encoder runs once over the whole thing, followed by a
   small decision transformer.
3. A scorer projects the hidden state **at each marker position** to one
   scalar. Softmax over marker positions gives the answer distribution.

So the output vocabulary is not a fixed head — it is *whatever options you put
in the prompt*. Three question types share the machinery: `choice`
(probabilities over named options), `score` (ordered rubric levels plus an
expected value), `noul` (P(true)). Temperature calibration is applied per
question type and option count.

Reported numbers on an M3 Max: **13.4 ms** P50 end-to-end for the 421M English
checkpoint, **7.4 ms** for the 322M multilingual one, **0 output tokens**,
943 MiB / 688 MiB peak allocation.

## 2. What Emera is

Emera plays a different game. A population of chaos-game organisms
("super-tokens") travels through a fixed token stream, and each one proposes
what comes next. There is no loss and no gradient. An organism pays to act,
earns a jackpot when it predicts correctly, and dies when its energy runs out.
Behaviour is meant to differentiate by selection rather than by design.

| | laya-mlx | emera |
|---|---|---|
| Objective | supervised typed decision | minimise `energy_spent / distance` |
| Learning | pretrained weights + RLCD upstream | selection, mutation, symbiogenesis |
| Credit assignment | gradient through one forward pass | energy ledger over a shared medium |
| Unit of knowledge | 421M dense parameters | a mortal organism with a genome |
| Output | calibrated probabilities over named options | a proposed byte segment |
| Interpretability | encoder hidden states | the population prints as text |
| Inference cost | one batched encoder pass, ~13 ms | K chaos rounds over live organisms |

### A finding worth stating plainly

In the current engine, the chaos dynamics are **not** what makes the
predictions. `_raw_decode_for_token` (`engine.py:1091`) calls only
`_decode_from_identity`, which aligns an organism's stored `identity_bytes`
against the cursor and proposes the literal continuation. The geometric decoder
`_decode_from_state` (`engine.py:1060`) — the one that projects an organism's
chaos state onto the token codebook — is never called on the proposal path.
`_infer_next_token` is likewise an energy-weighted vote over memorised byte
fragments matching the frontier byte.

The IFS, the gap field and the resonance reads still run, but on the prediction
path they only decide **who wakes up**; *what gets proposed* is a stored string.
Emera as it stands is, in effect, an evolutionary weighted-suffix model with a
chaos-driven attention mechanism bolted to the front.

That is not a criticism of the idea — it is the reason a classifier framing is
worth trying. Next-byte prediction has a copy shortcut: memorising a substring
of the world is always a locally winning move, so selection never has to build
anything else. A label is not a substring of the input. The shortcut does not
exist, so the readout has to do real work.

## 3. The proposed architecture

`classifier.py` keeps the substrate and swaps the game. An organism becomes a
**rule**:

> *a pattern it recognises · a label it leans toward · what it will stake on that*

### What is kept

- The chaos-game genome: IFS maps, `chaos_step`, emission amplitude and decay,
  resonance width, phase coupling, activation threshold, silence growth rate.
- The gap field as the only shared medium. Organisms never see the text
  directly; the example is projected into the gap as exogenous terrain and they
  feel it the same way they feel each other.
- K synchronised rounds per decision, so organisms can respond to what others
  emitted earlier in the same decision.
- A strictly conserved energy ledger. Energy only moves; it is never created.
  (Measured drift over 8,000 steps: ~1e-7 on a total of 9,000.)
- Death by depletion, silence credit for the dormant, reproduction paid for out
  of energy.

### What changes, and why

**The world becomes a labelled example stream.** Ground truth is one label per
step rather than a byte at a cursor. Credit assignment is now dense and exact.

**The readout becomes geometric.** Each label gets an orthogonal-at-init latent
from the same Hadamard scheme the base tokens use. An organism votes by
projecting `normalize(vote_bias + context_gain · state_vec)` onto that codebook
and taking the argmax, with the top-two margin as its confidence. This is the
same mechanism laya uses — score the options rather than generate the answer —
and it revives the decoder path the engine left dead. `vote_bias` is heredity;
`state_vec` is context, so the chaos dynamics modulate a vote that the genome
anchors.

**Payout becomes parimutuel.** Everyone who votes stakes energy. Losers' stakes
pay the winners in proportion to their own stake, minus a small rake. This is
exactly zero-sum, so the ledger stays conserved without a jackpot table, and it
prices each label by how crowded it is: being right when the crowd is wrong is
what pays. The Zipf division of labour the design doc wants falls out of the
payout rule instead of being encouraged by a tuned constant.

**Inactivity becomes abstention.** An organism that does not recognise the
example stakes nothing and collects a subsistence rebate. Narrow specialists
are viable — which is the whole point of a population.

**Discovery becomes a spawn.** When nobody staked the right label, the truth has
to be paid for: the system buys an organism that recognises this example, half
the time by reusing the pattern of whichever organism staked most on the wrong
answer (that pattern demonstrably fires here — only its lean was wrong). This is
the classifier's version of paying discovery cost and incorporating the revealed
token.

### Two places the book-world's machinery did not transfer

- **`_mutate_identity_bytes` composes parents by concatenation.** Sensible when
  a genome is a token sequence to be *proposed*; wrong when it is a detector.
  It produced organisms like `JudaJuda` and `DavidDavid`, which cannot occur in
  real text and can therefore only ever match their own prefix. Replaced with
  `_mutate_pattern`: trim (generalise), perturb, insert. Specialisation comes
  from discovery spawns cutting fresh patterns out of real examples instead.
- **Wealth is a noisy fitness signal.** Staking a fixed fraction of energy on a
  near-fair market is a multiplicative process, so survivors look rich whether
  or not they were right: an early run had a rule with a 7.3% hit rate (worse
  than the 16.7% chance baseline) among the richest organisms in the
  population. The readout now weights a vote by realised return on stake,
  `won_total / staked_total` — still nothing but the organism's own energy
  history, but it separates rules that earned their keep from ones that got
  lucky.

### Scaling

Cost per decision is proportional to the organisms an example *concerns*, not
to the population. Organisms are indexed by their pattern prefix; an example
wakes only those whose prefix occurs in it. Metabolism and silence credit run
on a slower clock (`economy_interval`, default 16 steps) with the tax compounded
in closed form, so the per-step sweep over dormant organisms is amortised
without changing any transfer. Together these took a 4,096-organism population
from intractable to ~20 ms/step.

## 4. Running it

```bash
python run_classify.py --task kjv-genre --steps 8000
python run_classify.py --task kjv-testament --steps 8000
python run_classify.py --task path/to/data.tsv --steps 8000   # label<TAB>text
```

Tasks are built from the KJV text already in `data/`: predict a verse's genre
(6 balanced classes), its testament (binary), or its book (66 classes). The
verse reference is stripped, so only the prose is visible. Splits are
deterministic by index hash.

## 5. Results

See `RESULTS.md` for measured numbers, baselines and sample populations.

## 6. Honest assessment

**What works.** The substrate transfers. Energy conservation holds exactly,
selection produces a readable population of rules, and accuracy is well above
chance and climbing with population size. The rules it finds are the ones a
person would write: `'ciples' → gospels` at 96% precision, `'Abraham ' → law`
at 82%.

**What does not, yet.** It is beaten by a hashed n-gram centroid baseline that
takes under a second to fit. The evolved population is a *weaker* ensemble of
*more interesting* rules. The gap between them is mostly coverage: a rule only
votes when its pattern occurs, so most of the population is silent on any given
example, and a decision rests on a few dozen votes.

**Where this is actually interesting.** Not as a way to get accuracy — a linear
model on character n-grams wins that, and a 322M-parameter encoder wins it by a
mile. It is interesting as a system that (a) produces a *readable, editable*
decision policy, (b) adds rules incrementally without retraining anything, and
(c) has a principled abstention story, since an organism that does not
recognise an example genuinely does not vote. Those are the three things a
dense encoder cannot give you, and they are worth something in exactly the
setting laya targets: a typed decision inside a piece of software, where you
have to explain why it chose what it chose.

**The comparison that would settle it.** Same typed-decision interface as laya
— `predict(state, {question: {type, criteria}})` — over the same questions, and
report accuracy, latency and abstention rate side by side. The interface is
already the same shape; it is the wrapper that is missing.

## 7. Next steps, in order of expected value

1. **Coverage.** Each organism votes on ~1% of examples. Either grow the
   population by another order of magnitude (the index supports it) or let one
   organism carry several patterns.
2. **Symbiogenesis.** Pair fusion is not yet implemented here. Two rules that
   are individually weak but jointly decisive — `'shall'` + `'LORD'` — are
   exactly what minting is for, and conjunctions are where a rule ensemble
   normally beats a bag of n-grams.
3. **Make resonance earn its place.** Wake is currently 80% literal pattern
   match. Measure accuracy as that weight goes to zero: if the gap field
   contributes nothing, either the coupling is wrong or the medium is not
   carrying information, and it is better to know which.
4. **The typed-decision wrapper.** `score` and `noul` are both trivial
   given the current readout: an ordered codebook for the former, two labels for
   the latter.
5. **Calibration.** The readout temperature is a constant. Laya calibrates per
   question type and option count; the same treatment would make the
   probabilities mean something.
