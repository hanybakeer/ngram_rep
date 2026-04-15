from src.data_prep.normalizer import Normalizer
import argparse
import dotenv  # type: ignore
dotenv.load_dotenv()
from src.normalizer import Normalizer # pyright: ignore[reportMissingImports]
from src.ngram_model import NGramModel # pyright: ignore[reportMissingImports]
import src.predictor # type: ignore



def dataprep():
    normalizer = Normalizer()
    normalizer.run()
    print("Dataprep complete: train_tokens.txt created.")


def model():
    model = NGramModel()
    model.train()
    model.save()
    print("Model complete: model.json and vocab.json created.")

def interface():
    normalizer = Normalizer()
    model = NGramModel()
    model.load()
    predictor = src.predictor.Predictor(normalizer, model)


