"""Labelled classification tasks for the Emera classifier substrate.

The book-world engine traverses an unlabelled token stream. A classifier needs
(text, label) pairs instead, so this module builds them from the same corpora
that already live in `data/`, plus a generic TSV loader for anything else.

Splits are deterministic: examples are assigned to train/val/test by a hash of
their index, so the same seed always yields the same partition regardless of
how many examples are requested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# KJV book -> genre grouping. Six classes, each with enough verses to learn from.
_KJV_GENRE: dict[str, str] = {}


def _register(genre: str, books: Iterable[str]) -> None:
    for book in books:
        _KJV_GENRE[book] = genre


_register("law", ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"])
_register(
    "history",
    [
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
        "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
    ],
)
_register(
    "wisdom",
    ["Job", "Psalm", "Proverbs", "Ecclesiastes", "Song of Solomon"],
)
_register(
    "prophets",
    [
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
        "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
        "Haggai", "Zechariah", "Malachi",
    ],
)
_register("gospels", ["Matthew", "Mark", "Luke", "John", "Acts"])
_register(
    "epistles",
    [
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
        "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
    ],
)

_NEW_TESTAMENT = {
    book for book, genre in _KJV_GENRE.items() if genre in {"gospels", "epistles"}
}


@dataclass(frozen=True)
class Example:
    text: str
    label: int


@dataclass
class Task:
    name: str
    labels: list[str]
    train: list[Example]
    val: list[Example]
    test: list[Example]

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def describe(self) -> str:
        counts = np.bincount(
            [ex.label for ex in self.train], minlength=self.num_labels
        )
        parts = ", ".join(
            f"{name}={int(n)}" for name, n in zip(self.labels, counts.tolist())
        )
        return (
            f"{self.name}: {self.num_labels} labels, "
            f"train={len(self.train)} val={len(self.val)} test={len(self.test)}\n"
            f"  train counts: {parts}"
        )

    def majority_baseline(self) -> float:
        counts = np.bincount(
            [ex.label for ex in self.train], minlength=self.num_labels
        )
        top = int(np.argmax(counts))
        hits = sum(1 for ex in self.test if ex.label == top)
        return float(hits) / max(len(self.test), 1)


def _split_bucket(index: int, salt: str) -> str:
    digest = hashlib.blake2b(
        f"{salt}:{index}".encode("utf-8"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _parse_kjv(path: Path) -> list[tuple[str, str]]:
    """Return (book, verse_text) pairs from the tab-separated KJV file."""
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "\t" not in line:
            continue
        ref, text = line.split("\t", 1)
        text = text.strip()
        if not text:
            continue
        # "1 Samuel 17:4" -> "1 Samuel"
        head = ref.rsplit(" ", 1)[0].strip()
        if head not in _KJV_GENRE:
            continue
        rows.append((head, text))
    return rows


def _balance(
    pool: dict[int, list[Example]], per_label: int, rng: np.random.Generator
) -> list[Example]:
    out: list[Example] = []
    for label in sorted(pool):
        items = pool[label]
        if per_label > 0 and len(items) > per_label:
            idx = rng.choice(len(items), size=per_label, replace=False)
            items = [items[int(i)] for i in sorted(idx.tolist())]
        out.extend(items)
    rng.shuffle(out)
    return out


def build_kjv_task(
    path: Path,
    kind: str = "genre",
    per_label_train: int = 1200,
    per_label_eval: int = 200,
    min_chars: int = 40,
    max_chars: int = 320,
    seed: int = 42,
) -> Task:
    """Classify a single KJV verse by genre, testament, or book.

    The verse reference is stripped, so only the prose is visible to the model.
    """
    rows = _parse_kjv(path)
    if not rows:
        raise ValueError(f"No KJV verses parsed from {path}")

    if kind == "genre":
        labels = ["law", "history", "wisdom", "prophets", "gospels", "epistles"]
        label_of = lambda book: _KJV_GENRE[book]  # noqa: E731
    elif kind == "testament":
        labels = ["old", "new"]
        label_of = lambda book: "new" if book in _NEW_TESTAMENT else "old"  # noqa: E731
    elif kind == "book":
        labels = sorted({book for book, _ in rows})
        label_of = lambda book: book  # noqa: E731
    else:
        raise ValueError("kind must be one of {'genre', 'testament', 'book'}")

    index = {name: i for i, name in enumerate(labels)}
    buckets: dict[str, dict[int, list[Example]]] = {
        "train": {}, "val": {}, "test": {}
    }
    for i, (book, text) in enumerate(rows):
        if len(text) < min_chars:
            continue
        if max_chars > 0:
            text = text[:max_chars]
        label = index[label_of(book)]
        split = _split_bucket(i, f"kjv-{kind}-{seed}")
        buckets[split].setdefault(label, []).append(Example(text=text, label=label))

    rng = np.random.default_rng(seed)
    return Task(
        name=f"kjv-{kind}",
        labels=labels,
        train=_balance(buckets["train"], per_label_train, rng),
        val=_balance(buckets["val"], per_label_eval, rng),
        test=_balance(buckets["test"], per_label_eval, rng),
    )


def build_tsv_task(
    path: Path,
    name: str | None = None,
    per_label_train: int = 0,
    per_label_eval: int = 0,
    max_chars: int = 320,
    seed: int = 42,
) -> Task:
    """Load a generic `label<TAB>text` file with the same deterministic splits."""
    label_names: list[str] = []
    index: dict[str, int] = {}
    buckets: dict[str, dict[int, list[Example]]] = {
        "train": {}, "val": {}, "test": {}
    }
    for i, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ):
        if "\t" not in line:
            continue
        raw_label, text = line.split("\t", 1)
        raw_label = raw_label.strip()
        text = text.strip()
        if not raw_label or not text:
            continue
        if max_chars > 0:
            text = text[:max_chars]
        if raw_label not in index:
            index[raw_label] = len(label_names)
            label_names.append(raw_label)
        label = index[raw_label]
        split = _split_bucket(i, f"tsv-{path.name}-{seed}")
        buckets[split].setdefault(label, []).append(Example(text=text, label=label))

    if not label_names:
        raise ValueError(f"No `label<TAB>text` rows parsed from {path}")

    rng = np.random.default_rng(seed)
    return Task(
        name=name or path.stem,
        labels=label_names,
        train=_balance(buckets["train"], per_label_train, rng),
        val=_balance(buckets["val"], per_label_eval, rng),
        test=_balance(buckets["test"], per_label_eval, rng),
    )


def load_task(
    spec: str,
    data_dir: Path = Path("data"),
    per_label_train: int = 1200,
    per_label_eval: int = 200,
    seed: int = 42,
) -> Task:
    """Resolve a task spec: `kjv-genre`, `kjv-testament`, `kjv-book`, or a TSV path."""
    if spec.startswith("kjv-"):
        kind = spec.split("-", 1)[1]
        path = data_dir / "bible.txt"
        if not path.exists():
            path = data_dir / "tiny_bible.txt"
        return build_kjv_task(
            path,
            kind=kind,
            per_label_train=per_label_train,
            per_label_eval=per_label_eval,
            seed=seed,
        )
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Unknown task spec and not a file: {spec}")
    return build_tsv_task(
        path,
        per_label_train=per_label_train,
        per_label_eval=per_label_eval,
        seed=seed,
    )


def centroid_baseline(task: Task, ngram: int = 4, buckets: int = 2**16) -> float:
    """Hashed character n-gram nearest-centroid baseline (tf-idf-ish, no training loop).

    Cheap reference point so the evolved population's accuracy has context.
    """
    def featurize(text: str) -> tuple[np.ndarray, np.ndarray]:
        raw = text.lower().encode("utf-8", errors="ignore")
        if len(raw) < ngram:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(arr, ngram)
        h = np.zeros(windows.shape[0], dtype=np.int64)
        for k in range(ngram):
            h = (h * 257 + windows[:, k]) % buckets
        keys, counts = np.unique(h, return_counts=True)
        return keys, counts.astype(np.float32)

    n_labels = task.num_labels
    df = np.zeros((buckets,), dtype=np.float32)
    sums = np.zeros((n_labels, buckets), dtype=np.float32)
    for ex in task.train:
        keys, counts = featurize(ex.text)
        if keys.size == 0:
            continue
        sums[ex.label, keys] += counts
        df[keys] += 1.0

    idf = np.log((len(task.train) + 1.0) / (df + 1.0)).astype(np.float32)
    centroids = sums * idf[None, :]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)

    hits = 0
    for ex in task.test:
        keys, counts = featurize(ex.text)
        if keys.size == 0:
            continue
        vec = counts * idf[keys]
        vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
        scores = centroids[:, keys] @ vec
        if int(np.argmax(scores)) == ex.label:
            hits += 1
    return float(hits) / max(len(task.test), 1)


def texts_and_labels(examples: Sequence[Example]) -> tuple[list[str], np.ndarray]:
    return (
        [ex.text for ex in examples],
        np.asarray([ex.label for ex in examples], dtype=np.int32),
    )
