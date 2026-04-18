"""
Streamlit UI (Module 6.5 — Extra Credit)

This UI runs alongside the CLI (it does not replace it).
Run with:
    streamlit run src/ui/app.py

Implements PredictorUI:
- Loads config/.env
- Loads the pre-built model + vocab
- Accepts user input
- Calls Predictor.predict_next()
- Displays Top-K predictions in the browser
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


class PredictorUI:
    """
    Browser-based Streamlit UI for the next-word predictor.

    Responsibility:
        - Load configuration (.env), model.json, vocab.json.
        - Provide a simple interface to enter text and display Top-K next-word predictions.
        - Keep UI separate from CLI; CLI remains available via main.py --step inference.

    Usage:
        streamlit run src/ui/app.py
    """

    def __init__(self) -> None:
        """
        Initialize the UI by loading environment variables and preparing paths.
        Model loading is done lazily (cached) in _load_dependencies().

        Returns:
            None
        """
        project_root = Path(__file__).resolve().parents[2]  # .../project-root
        load_dotenv(dotenv_path=project_root / "config" / ".env")

        self.model_path = os.getenv("MODEL", "data/model/model.json")
        self.vocab_path = os.getenv("VOCAB", "data/model/vocab.json")

        self.default_top_k = int(os.getenv("TOP_K", "3"))
        self.max_top_k = int(os.getenv("MAX_TOP_K", "10"))  # optional, safe default
        self.ngram_order = int(os.getenv("NGRAM_ORDER", "4"))
        self.unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_dependencies(model_path: str, vocab_path: str, ngram_order: int, unk_threshold: int) -> Predictor:
        """
        Load the model + vocab once (cached by Streamlit), then construct Predictor.

        Parameters:
            model_path: Path to model.json
            vocab_path: Path to vocab.json
            ngram_order: NGRAM_ORDER from env
            unk_threshold: UNK_THRESHOLD from env

        Returns:
            Predictor instance (ready for inference)
        """
        # Validate files exist
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing model file: {model_path}. Build it with: python main.py --step model")
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}. Build it with: python main.py --step model")

        normalizer = Normalizer()

        model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)
        model.load(model_path, vocab_path)

        predictor = Predictor(model, normalizer)
        return predictor

    @staticmethod
    def _normalize_predictions(predictions: Any) -> List[Tuple[str, Union[float, None]]]:
        """
        Make UI robust to different Predictor.predict_next() return formats.

        Supported formats:
            - List[str]
            - List[Tuple[word, prob]]
            - Dict[word] = prob
            - Anything else -> best-effort string conversion

        Parameters:
            predictions: output returned by Predictor.predict_next()

        Returns:
            List of (word, prob_or_none)
        """
        if predictions is None:
            return []

        # Dict[str, float]
        if isinstance(predictions, dict):
            # preserve sorted order by prob (desc) if not already sorted
            items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return [(w, float(p)) for w, p in items]

        # List[...]
        if isinstance(predictions, list):
            if not predictions:
                return []
            first = predictions[0]
            # List[Tuple[word, prob]]
            if isinstance(first, (tuple, list)) and len(first) >= 2:
                out = []
                for item in predictions:
                    try:
                        w = str(item[0])
                        p = float(item[1]) if item[1] is not None else None
                        out.append((w, p))
                    except Exception:
                        out.append((str(item), None))
                return out
            # List[str]
            return [(str(x), None) for x in predictions]

        # Fallback
        return [(str(predictions), None)]

    def render(self) -> None:
        """
        Render the Streamlit UI: sidebar config, input box, predictions output.

        Returns:
            None
        """
        st.set_page_config(page_title="N-Gram Next-Word Predictor", page_icon="🧠", layout="centered")

        st.title("🧠 N‑Gram Next‑Word Predictor")
        st.caption("Streamlit UI (Extra Credit) — runs alongside the CLI, not instead of it.")

        with st.sidebar:
            st.header("Settings")
            st.write("Model / vocab paths (from .env):")
            st.code(f"MODEL={self.model_path}\nVOCAB={self.vocab_path}", language="text")

            top_k = st.slider("Top‑K predictions", min_value=1, max_value=self.max_top_k, value=self.default_top_k)

            st.divider()
            st.subheader("Actions")
            reload_clicked = st.button("🔄 Reload model", use_container_width=True)
            st.caption("If you rebuild model.json/vocab.json, reload here.")

        # Reload: clear cache and re-load on next call
        if reload_clicked:
            st.cache_resource.clear()
            st.success("Cache cleared. Model will reload on next prediction.")

        # Load predictor (cached)
        try:
            predictor = self._load_dependencies(
                self.model_path, self.vocab_path, self.ngram_order, self.unk_threshold
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.subheader("Enter text")
        user_input = st.text_input("Type a few words, then press Predict:", value="", placeholder="e.g., holmes said to")

        col1, col2 = st.columns([1, 1])
        with col1:
            predict_clicked = st.button("✨ Predict", use_container_width=True)
        with col2:
            clear_clicked = st.button("🧹 Clear", use_container_width=True)

        if clear_clicked:
            # Streamlit doesn't allow direct clearing of text_input without session_state
            st.session_state["__clear__"] = True
            st.rerun()

        # Optional: clear hack with session state
        if st.session_state.get("__clear__", False):
            st.session_state["__clear__"] = False
            # Rerun already occurred; just don't display predictions
            return

        if predict_clicked:
            if not user_input.strip():
                st.warning("Please enter some text.")
                st.stop()

            with st.spinner("Predicting..."):
                try:
                    raw_predictions = predictor.predict_next(user_input, top_k)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            preds = self._normalize_predictions(raw_predictions)

            st.subheader("Predictions")
            if not preds:
                st.info("No predictions available for this input.")
                st.stop()

            # Display: table if probs exist, otherwise bullet list
            has_probs = any(p is not None for _, p in preds)

            if has_probs:
                # Show as a neat table
                st.write("")
                st.dataframe(
                    [{"word": w, "probability": p} for w, p in preds],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("")
                st.write(", ".join([w for w, _ in preds]))

        st.divider()
        st.caption(
            "CLI remains available via: `python main.py --step inference`  •  "
            "Build model via: `python main.py --step model`"
        )


def main() -> None:
    """
    Streamlit entrypoint.
    """
    ui = PredictorUI()
    ui.render()


if __name__ == "__main__":
    main()