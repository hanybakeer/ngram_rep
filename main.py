"""
main.py

Single entry point for the n-gram next-word predictor project.
This file wires together data preparation, model building, and inference CLI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel
import logging

# Configure to show messages at INFO level and above
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logging.debug("This will not be shown")  # Level 10
logging.info("Application started")      # Level 20
logging.warning("Something is odd")      # Level 30

def run_dataprep(normalizer: Normalizer, train_raw_dir: str, train_tokens_path: str) -> None:
    """
    Run the data preparation pipeline.

    Parameters:
        normalizer (Normalizer): Shared normalizer instance.
        train_raw_dir (str): Folder containing raw training text files.
        train_tokens_path (str): Output path for tokenized training sentences.

    Returns:
        None
    """
    tokenized_sentences = normalizer.prepare_corpus(train_raw_dir)
    normalizer.save(tokenized_sentences, train_tokens_path)
    print(f"Data preparation complete: {train_tokens_path}")


def run_model(
    model: NGramModel,
    train_tokens_path: str,
    model_path: str,
    vocab_path: str
) -> None:
    """
    Build vocabulary and n-gram probability tables, then save them.

    Parameters:
        model (NGramModel): N-gram model instance.
        train_tokens_path (str): Path to tokenized training data.
        model_path (str): Output path for model JSON.
        vocab_path (str): Output path for vocabulary JSON.

    Returns:
        None
    """
    model.build_vocab(train_tokens_path)
    model.build_counts_and_probabilities(train_tokens_path)
    model.save_model(model_path)
    model.save_vocab(vocab_path)

    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")


def run_inference(
    predictor: Predictor,
    top_k: int
) -> None:
    """
    Start the interactive CLI prediction loop.

    Parameters:
        predictor (Predictor): Predictor instance.
        top_k (int): Number of predictions to print.

    Returns:
        None
    """
    print("Interactive next-word predictor")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("> ").strip()

            if user_input.lower() == "quit":
                print("Goodbye.")
                break

            predictions = predictor.predict_next(user_input, top_k)
            print(f"Predictions: {predictions}")

    except KeyboardInterrupt:
        print("\nGoodbye.")


def ensure_model_files_exist(model_path: str, vocab_path: str) -> None:
    """
    Ensure saved model files exist before inference.

    Parameters:
        model_path (str): Path to model JSON.
        vocab_path (str): Path to vocabulary JSON.

    Returns:
        None

    Raises:
        FileNotFoundError: If model or vocab file does not exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")


def main() -> None:
    """
    Load configuration, parse arguments, instantiate dependencies, and run
    the requested pipeline step.

    Returns:
        None
    """
    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / "config" / ".env")

    parser = argparse.ArgumentParser(description="N-Gram Next-Word Predictor")
    parser.add_argument(
        "--step",
        required=True,
        choices=["dataprep", "model", "inference", "all"],
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    train_raw_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))
    top_k = int(os.getenv("TOP_K", "3"))
    ngram_order = int(os.getenv("NGRAM_ORDER", "4"))

    if not all([train_raw_dir, train_tokens, model_path, vocab_path]):
        raise ValueError("Missing required environment variables in config/.env")

    normalizer = Normalizer()
    ngram_model = NGramModel(
        ngram_order=ngram_order,
        unk_threshold=unk_threshold
    )

    if args.step == "dataprep":
        run_dataprep(normalizer, train_raw_dir, train_tokens)

    elif args.step == "model":
        run_model(ngram_model, train_tokens, model_path, vocab_path)

    elif args.step == "inference":
        ensure_model_files_exist(model_path, vocab_path)
        ngram_model.load(model_path, vocab_path)
        predictor = Predictor(ngram_model, normalizer)
        run_inference(predictor, top_k)

    elif args.step == "all":
        run_dataprep(normalizer, train_raw_dir, train_tokens)
        run_model(ngram_model, train_tokens, model_path, vocab_path)
        ngram_model.load(model_path, vocab_path)
        predictor = Predictor(ngram_model, normalizer)
        run_inference(predictor, top_k)


if __name__ == "__main__":
    main()