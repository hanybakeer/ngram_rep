
"""
Predictor (Module 3 — Inference)

Responsibility:
- Accept a pre-loaded NGramModel and Normalizer via the constructor.
- Normalize user input text.
- Extract the last NGRAM_ORDER - 1 words as context.
- Perform backoff lookup via NGramModel.lookup().
- Return the top-k predicted next words sorted by probability.

Backoff logic is delegated entirely to NGramModel.lookup().
This module does not load models, vocabularies, or files.
"""

from typing import List
import os
import sys
from dotenv import load_dotenv

try:
    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel, UNK_TOKEN
except ImportError:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    src_root = os.path.join(repo_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from data_prep.normalizer import Normalizer
    from model.ngram_model import NGramModel, UNK_TOKEN


class Predictor:
    """
    Inference layer for next-word prediction using a trained NGramModel.
    """

    def __init__(self, model: NGramModel, normalizer: Normalizer, top_k: int | None = None):
        self.model = model
        self.normalizer = normalizer
        self.ngram_order = model.ngram_order
        self.top_k = top_k if top_k is not None else self._load_top_k_from_env()

    @staticmethod
    def _load_top_k_from_env() -> int:
        load_dotenv(dotenv_path="config/.env")
        try:
            return int(os.getenv("TOP_K", 5))
        except (TypeError, ValueError):
            return 5

    # -------------------------------------------------
    # Step 01 — Normalize input and extract context
    # -------------------------------------------------
    def normalize(self, text: str) -> List[str]:
        """
        Normalize input text and extract the last NGRAM_ORDER - 1 words.

        Parameters:
        text (str): Raw user input

        Returns:
        List[str]: Context words (length <= NGRAM_ORDER - 1)
        """
        normalized_text = self.normalizer.normalizer(text)
        tokens = normalized_text.split()

        return tokens[-(self.ngram_order - 1):]

    # -------------------------------------------------
    # Step 02 — Map OOV words to <UNK>
    # -------------------------------------------------
    def map_oov(self, context: List[str]) -> List[str]:
        """
        Replace out-of-vocabulary words with <UNK>.

        Parameters:
        context (List[str]): Context tokens

        Returns:
        List[str]: Context with OOV words mapped to <UNK>
        """
        return [
            word if word in self.model.vocab_set else UNK_TOKEN
            for word in context
        ]

    # -------------------------------------------------
    # Steps 03–04 — Backoff lookup and top-k ranking
    # -------------------------------------------------
    def predict_next(self, text: str, k: int | None = None) -> List[str]:
        """
        Predict the top-k most likely next words for a given input string.

        Algorithm:
        normalize → map_oov → model.lookup() → rank → return top-k words

        Parameters:
        text (str): Raw user input
        k (int, optional): Number of predictions to return

        Returns:
        List[str]: Top-k predicted next words sorted by probability
        """
        if k is None:
            k = self.top_k

        # Step 01
        context = self.normalize(text)

        # Step 02
        context = self.map_oov(context)

        # Step 03 (backoff handled by model)
        probabilities = self.model.lookup(context)

        if not probabilities:
            return []

        # Step 04 — rank and return
        sorted_words = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [word for word, _ in sorted_words[:k]]

"""
if __name__ == "__main__":
    load_dotenv(dotenv_path="config/.env")
    model = NGramModel.from_env()
    model.load("data/model/model.json", "data/model/vocab.json")
    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)
    print(predictor.predict_next("this is"))
"""
