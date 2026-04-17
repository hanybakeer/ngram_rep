import os
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

class Normalizer:
    """
    Load, clean, normalize, tokenize, and save corpus text.

    This class is used in two contexts:
    1. Data preparation on full raw text files.
    2. Inference-time normalization on a single user input string.
    """

    def load(self,folder_path: str) -> str:
        """
        Load all .txt files in a folder.
        Parameters: folder_path (str): Path to the folder containing .txt files.
        Returns:str: The raw text loaded from the file.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"
        return text 
    def strip_gutenberg(self,in_text:str)->str:
        """
        This method removes all headers and footers from the text
        Parameters: text (str)
        Returns: str without header and footer
        """
        pattern = re.compile(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*\s*(.*?)\s*\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",re.DOTALL)
        books = pattern.findall(in_text)
        return "\n\n".join(book.strip() for book in books)
    
    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.

        Parameters:
            text (str): Input text.

        Returns:
            str: Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text while keeping letters, digits, and spaces.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text without punctuation.
        """
        return re.sub(r"[^\w\s]", " ", text)

    def remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text without digits.
        """
        return re.sub(r"\d+", " ", text)

    def remove_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and blank lines.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text with normalized whitespace.
        """
        return re.sub(r"\s+", " ", text).strip()


    def normalize(self, text: str) -> str:
        """
        Apply lowercase, remove punctuation, remove numbers, and normalize whitespace.

        Parameters:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text
    def sentence_tokenize(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Parameters:
            text (str): Input text.

        Returns:
            List[str]: List of sentence strings.
        """
        try:
            return sent_tokenize(text)
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")
            return sent_tokenize(text)
    def word_tokenize(self,sentence: list[str]) -> list[str]:
        """
        Split a normalized sentence into tokens.

        Parameters:
            sentence (str): A single sentence string.

        Returns:
            List[str]: List of word tokens.
        """
        if not sentence.strip():
            return []
        return sentence.split()
    def prepare_corpus(self, folder_path: str) -> List[List[str]]:
        """
        Run the full data-preparation pipeline on all files in a folder.

        To preserve sentence boundaries, sentence tokenization is done on
        stripped raw text first, then each sentence is normalized using the
        same normalize() method used during inference.

        Parameters:
            folder_path (str): Folder containing raw .txt files.

        Returns:
            List[List[str]]: Tokenized sentences, one list of tokens per sentence.
        """
        raw_text = self.load(folder_path)
        stripped_text = self.strip_gutenberg(raw_text)
        raw_sentences = self.sentence_tokenize(stripped_text)

        tokenized_sentences: List[List[str]] = []
        for sentence in raw_sentences:
            normalized_sentence = self.normalize(sentence)
            tokens = self.word_tokenize(normalized_sentence)
            if tokens:
                tokenized_sentences.append(tokens)

        return tokenized_sentences
    def save(self, sentences: List[List[str]], filepath: str) -> None:
        """
        Save tokenized sentences to a file, one sentence per line.

        Parameters:
            sentences (List[List[str]]): Tokenized sentences.
            filepath (str): Output file path.

        Returns:
            None
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for sentence_tokens in sentences:
                file.write(" ".join(sentence_tokens) + "\n")
 

if __name__=="__main__":
    #   load_dotenv(dotenv_path='config/.env')

    #   tokenized_sentences = Normalizer().prepare_corpus(os.getenv("TRAIN_RAW_DIR"))
    #   Normalizer().save(tokenized_sentences, os.getenv("TRAIN_TOKENS"))
    #   print(f"Data preparation complete:")
    pass