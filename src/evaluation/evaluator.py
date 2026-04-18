"""
Evaluator (Module 9.5 — Model Evaluator)

Responsibility:
- Compute perplexity on a held-out evaluation corpus using a pre-loaded NGramModel.
- For each word w_i in the evaluation corpus:
    * get P(w_i | context) via NGramModel.lookup() (which implements backoff)
    * accumulate log2 P(w_i | context)
- Cross-entropy: H = - (1/N) * Σ log2 P(w_i | context)
- Perplexity: 2^H
Where N is the number of evaluated words (skipped words excluded).

Skipping:
- If P(w_i | context) is zero at all orders (not found in lookup distribution), skip that word.

Warning:
- Print a warning if more than 20% of tokens are skipped.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

UNK_TOKEN = "<UNK>"


class Evaluator:
    """
    Compute perplexity for a pre-loaded NGramModel on a held-out corpus.

    Methods:
        __init__(model, normalizer):
            Store references to a loaded NGramModel and a Normalizer.

        score_word(word, context):
            Return log2 P(word | context) using NGramModel.lookup().
            Return None if probability is zero (missing) at all orders.

        compute_perplexity(eval_file):
            Compute perplexity over a full eval corpus (file or folder).
            Print a warning if more than 20% words are skipped.

        run(eval_file):
            Orchestrate compute_perplexity() and print result.
    """

    def __init__(self, model, normalizer):
        """
        Parameters:
            model: Pre-loaded NGramModel instance. Must have vocab/model loaded.
            normalizer: Normalizer instance (your provided implementation).

        Returns:
            None
        """
        self.model = model
        self.normalizer = normalizer

        # Determine max context length = NGRAM_ORDER - 1
        self.ngram_order = getattr(model, "ngram_order", None)
        if not self.ngram_order:
            # Fallback: infer from model keys "1gram", "2gram", ...
            keys = getattr(model, "model", {}) or {}
            orders = []
            for k in keys:
                if isinstance(k, str) and k.endswith("gram"):
                    try:
                        orders.append(int(k[:-4]))
                    except ValueError:
                        pass
            self.ngram_order = max(orders) if orders else 1

        self.max_context_len = max(0, int(self.ngram_order) - 1)

        # Cache vocab set if not present on model
        self.vocab_set = getattr(model, "vocab_set", None)
        if self.vocab_set is None:
            vocab = getattr(model, "vocab", None) or []
            self.vocab_set = set(vocab)

    def _tokenize_eval(self, eval_path: str) -> List[List[str]]:
        """
        Tokenize the evaluation corpus using the SAME pipeline as training (Normalizer).

        Supports:
          - eval_path as a folder containing one or more .txt files:
                Normalizer.prepare_corpus(folder)
          - eval_path as a single .txt file:
                read file -> strip_gutenberg -> sentence_tokenize -> normalize -> word_tokenize

        Parameters:
            eval_path: File path or folder path.

        Returns:
            List of sentences; each sentence is a List[str] tokens.
        """
        p = Path(eval_path)

        if p.is_dir():
            # Your Normalizer already loads all .txt in the folder and preserves sentence boundaries
            return self.normalizer.prepare_corpus(str(p))

        if p.is_file():
            raw_text = p.read_text(encoding="utf-8")
            stripped = self.normalizer.strip_gutenberg(raw_text)
            raw_sentences = self.normalizer.sentence_tokenize(stripped)

            tokenized: List[List[str]] = []
            for s in raw_sentences:
                normalized = self.normalizer.normalize(s)
                tokens = self.normalizer.word_tokenize(normalized)
                if tokens:
                    tokenized.append(tokens)
            return tokenized

        raise FileNotFoundError(f"Evaluation path not found: {eval_path}")

    def score_word(self, word: str, context: List[str]) -> Optional[float]:
        """
        Score a single word w_i given its context, using backoff via model.lookup().

        Parameters:
            word: The target word/token w_i.
            context: History tokens (previous tokens), typically length <= NGRAM_ORDER-1.

        Returns:
            log2(P(word | context)) if probability > 0, else None if zero probability.
        """
        # Map OOV target to <UNK> so evaluation can still score unknowns as <UNK>
        target = word if word in self.vocab_set else UNK_TOKEN

        dist: Dict[str, float] = self.model.lookup(context)
        if not dist:
            return None

        p = dist.get(target, 0.0)
        if p <= 0.0:
            return None

        return math.log2(p)

    def compute_perplexity(self, eval_file: str) -> Tuple[float, int, int]:
        """
        Compute perplexity over the full evaluation corpus.

        Steps:
          - Tokenize eval corpus using Normalizer (same pipeline as training)
          - For each token w_i:
              * context is previous (NGRAM_ORDER-1) tokens within the sentence
              * accumulate log2 P(w_i|context) if available; else skip
          - Cross-entropy H = - (1/N) * sum_log2
          - Perplexity = 2^H
          - Warn if skipped tokens > 20% of total

        Parameters:
            eval_file: Path to evaluation file OR folder containing .txt files.

        Returns:
            (perplexity, words_evaluated, words_skipped)
        """
        sentences = self._tokenize_eval(eval_file)

        sum_log2_p = 0.0
        evaluated = 0
        skipped = 0
        total = 0

        for sent in sentences:
            history: List[str] = []

            for w in sent:
                total += 1
                context = history[-self.max_context_len:] if self.max_context_len > 0 else []

                s = self.score_word(w, context)
                if s is None:
                    skipped += 1
                else:
                    sum_log2_p += s
                    evaluated += 1

                history.append(w)

        if evaluated == 0:
            raise ValueError(
                "No words were evaluated (all tokens had zero probability). "
                "Likely causes: normalization mismatch, wrong vocab/model, or overly small vocab."
            )

        H = - (sum_log2_p / evaluated)       # cross-entropy in bits
        perplexity = 2 ** H

        skip_ratio = skipped / max(1, total)
        if skip_ratio > 0.20:
            print(
                f"Warning: {skip_ratio:.1%} of tokens were skipped due to zero probability. "
                "This often indicates a normalization mismatch between train and eval."
            )

        return perplexity, evaluated, skipped

    def run(self, eval_file: str) -> None:
        """
        Run evaluation and print results.

        Parameters:
            eval_file: Path to evaluation file or folder.

        Returns:
            None
        """
        perplexity, evaluated, skipped = self.compute_perplexity(eval_file)

        # Expected output style
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Words evaluated: {evaluated:,}")
        print(f"Words skipped (zero probability): {skipped:,}")