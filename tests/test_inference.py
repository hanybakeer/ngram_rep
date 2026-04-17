
import sys
import os
import unittest

# -------------------------------------------------
# Make project root importable
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Now imports work as expected
# -------------------------------------------------
from model.ngram_model import NGramModel
from data_prep.normalizer import Normalizer
from inference.predictor import Predictor


class TestPredictor(unittest.TestCase):
    """
    Tests for Module 3 — Inference (Predictor)
    """

    @classmethod
    def setUpClass(cls):
        """
        Load model and predictor once for all tests.
        """
        cls.model = NGramModel.from_env()
        cls.model.load(
            "data/model/model.json",
            "data/model/vocab.json"
        )

        cls.normalizer = Normalizer()
        cls.predictor = Predictor(cls.model, cls.normalizer)

    # -------------------------------------------------
    # Core functionality
    # -------------------------------------------------
    def test_returns_list(self):
        result = self.predictor.predict_next("sherlock holmes")
        self.assertIsInstance(result, list)

    def test_non_empty_for_common_input(self):
        result = self.predictor.predict_next("sherlock holmes")
        self.assertGreater(len(result), 0)

    # -------------------------------------------------
    # TOP-K behavior
    # -------------------------------------------------
    def test_respects_top_k(self):
        k = self.predictor.top_k
        result = self.predictor.predict_next("sherlock holmes")
        self.assertLessEqual(len(result), k)

    # -------------------------------------------------
    # Edge cases
    # -------------------------------------------------
    def test_empty_input(self):
        result = self.predictor.predict_next("")
        self.assertIsInstance(result, list)

    def test_oov_input(self):
        result = self.predictor.predict_next("qwerty asdf zxcv")
        self.assertIsInstance(result, list)

    # -------------------------------------------------
    # Normalization consistency
    # -------------------------------------------------
    def test_case_insensitive(self):
        r1 = self.predictor.predict_next("Sherlock Holmes")
        r2 = self.predictor.predict_next("sherlock holmes")
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
