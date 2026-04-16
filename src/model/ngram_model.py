"""
NGramModel (Module 6.2 — Model)

Responsibility:
- Build and store n-gram probability tables (MLE) for all orders from 1 up to NGRAM_ORDER.
- Provide a single backoff lookup() method that tries the highest order first and falls back
  to lower orders down to unigrams (stupid backoff without smoothing/discounting).
- Save/load model.json and vocab.json.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from dotenv import load_dotenv


UNK_TOKEN = "<UNK>"


def _ensure_dir_for_file(file_path: str) -> None:
    """
    Create parent directory for a file path if it doesn't exist.
    """
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_tokenized_sentences(token_file: str):
    """
    Read train_tokens.txt with one tokenized sentence per line.
    """
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()


@dataclass
class NGramModel:
    """
    Builds, stores, and exposes n-gram probability tables and backoff lookup
    across all orders from 1 up to NGRAM_ORDER.
    """

    ngram_order: int
    unk_threshold: int

    vocab: List[str] = None
    model: Dict[str, dict] = None

    def __post_init__(self):
        self.vocab = self.vocab or []
        self.model = self.model or {}
        self.vocab_set = set(self.vocab)

    # -------------------------------------------------
    # Step 01 — Build vocabulary with UNK_THRESHOLD
    # -------------------------------------------------
    def build_vocab(self, token_file: str) -> List[str]:
        """
        Build vocabulary from token file and apply UNK_THRESHOLD.
        """
        word_counts = Counter()
        for tokens in _iter_tokenized_sentences(token_file):
            word_counts.update(tokens)

        vocab = [w for w, c in word_counts.items() if c >= self.unk_threshold]

        if UNK_TOKEN not in vocab:
            vocab.append(UNK_TOKEN)

        vocab = sorted(vocab)
        self.vocab = vocab
        self.vocab_set = set(vocab)
        return vocab

    # -------------------------------------------------
    # Steps 02–04 — Counts, MLE probabilities, OOV→UNK
    # -------------------------------------------------
    def build_counts_and_probabilities(self, token_file: str) -> Dict[str, dict]:
        """
        Count all n-grams (1..NGRAM_ORDER) and compute MLE probabilities.
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        unigram_counts = Counter()
        total_unigrams = 0

        context_counts = {k: Counter() for k in range(2, self.ngram_order + 1)}
        next_word_counts = {
            k: defaultdict(Counter) for k in range(2, self.ngram_order + 1)
        }

        for tokens in _iter_tokenized_sentences(token_file):
            tokens = [t if t in self.vocab_set else UNK_TOKEN for t in tokens]

            unigram_counts.update(tokens)
            total_unigrams += len(tokens)

            for k in range(2, self.ngram_order + 1):
                if len(tokens) < k:
                    continue

                for i in range(len(tokens) - k + 1):
                    ngram = tokens[i : i + k]
                    context = tuple(ngram[:-1])
                    word = ngram[-1]

                    context_counts[k][context] += 1
                    next_word_counts[k][context][word] += 1

        model = {}

        # 1-gram
        model["1gram"] = {
            w: unigram_counts[w] / total_unigrams
            for w in unigram_counts
        }

        # higher orders
        for k in range(2, self.ngram_order + 1):
            table = {}
            for context, ctx_count in context_counts[k].items():
                ctx_key = " ".join(context)
                table[ctx_key] = {
                    w: c / ctx_count
                    for w, c in next_word_counts[k][context].items()
                }
            model[f"{k}gram"] = table

        self.model = model
        return model

    # -------------------------------------------------
    # Backoff lookup (single source of logic)
    # -------------------------------------------------
    def lookup(self, context: List[str]) -> Dict[str, float]:
        """
        Try highest-order context first; fall back down to unigram.
        """
        if not self.model:
            return {}

        context = [w if w in self.vocab_set else UNK_TOKEN for w in context]

        for order in range(self.ngram_order, 1, -1):
            ctx_len = order - 1
            if len(context) < ctx_len:
                continue

            ctx = " ".join(context[-ctx_len:])
            table = self.model.get(f"{order}gram", {})
            if ctx in table:
                return table[ctx]

        return self.model.get("1gram", {})

    # -------------------------------------------------
    # Save / load
    # -------------------------------------------------
    def save_model(self, path: str) -> None:
        _ensure_dir_for_file(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model, f, indent=2, ensure_ascii=False)

    def save_vocab(self, path: str) -> None:
        _ensure_dir_for_file(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    def load(self, model_path: str, vocab_path: str) -> None:
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.vocab_set = set(self.vocab)

    # -------------------------------------------------
    # Factory: load config/.env via load_dotenv ✅
    # -------------------------------------------------
    @classmethod
    def from_env(cls) -> "NGramModel":
        """
        Initialize model using values from config/.env.
        """
        load_dotenv(dotenv_path="config/.env")

        ngram_order = int(os.getenv("NGRAM_ORDER", 4))
        unk_threshold = int(os.getenv("UNK_THRESHOLD", 1))

        return cls(
            ngram_order=ngram_order,
            unk_threshold=unk_threshold,
        )


# -------------------------------------------------
# Optional entry point
# -------------------------------------------------
def main():
    token_file = "data/processed/train_tokens.txt"
    model_out = "data/model/model.json"
    vocab_out = "data/model/vocab.json"

    model = NGramModel.from_env()
    model.build_vocab(token_file)
    model.build_counts_and_probabilities(token_file)
    model.save_model(model_out)
    model.save_vocab(vocab_out)


if __name__ == "__main__":
    main()