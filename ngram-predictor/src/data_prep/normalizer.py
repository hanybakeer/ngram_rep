# add main() if __name__=="__main__"

import os
from pydoc import text
import re
import string
#from dotenv import load_dotenv


class Normalizer:
    """
    This class describes the module's responsibility: loading, cleaning, tokenizing, and saving the corpus.

    Parameters:
    Text file

    Returns:
    tokenized file

    """    
    def load(self,folder_path: str) -> str:
        """
        Load all .txt files in a folder.
        Parameters: folder_path (str): Path to the folder containing .txt files.
        Returns:str: The raw text loaded from the file.
        """
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
    def normalizer(self,text:str)->str:
        """
        This method applies lowercase, remove punctuation, remove numbers, and remove extra whitespace in order.
        Parameters: text (str)
        Returns: str normalized
        """
        #lowercase
        text = text.lower()

        #remove punctuation, but keep . ! ?
        punctuation_to_remove = string.punctuation.replace('.', '').replace('!', '').replace('?', '')
        text = text.translate(str.maketrans('', '', punctuation_to_remove))
        
        #remove numbers
        text = re.sub(r'\d+', '', text)
        
        #remove extra whitespace
        text = re.sub(r'[ \t]+', ' ', text)

        #remove spaces before sentence punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)

        return text
    def sentence_tokenize(self,text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]
    def word_tokenize(self,sentences: list[str]) -> list[str]:
        #return [word for sentence in sentences for word in sentence.split()]
        return [word for word in sentences.split()]
    def write_list_to_file(self,items_sentence: list[str],items_word: list[str], output_file: str)->None:
        """
        This method write list to file
        Parameters: text (str)
        Returns: file written
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for item_sentence in items_sentence:
                f.write(item_sentence + "\n")
            #for item_word in items_word:
             #   f.write(item_word + "")
        return text   
#tests

#"""
#works for hany
TRAIN_RAW_DIR="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\raw\\train"
EVAL_RAW_DIR="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\raw\eval"
TRAIN_TOKENS="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\processed\\train_tokens.txt"
EVAL_TOKENS="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\processed\\eval_tokens.txt"
MODEL="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\model\model.json"
VOCAB="D:\\AI\Python\practice\\ngram_capstone\\ngram_rep\\ngram-predictor\data\\model\vocab.json"
#Testing branch
#"""


"""
#works for salma
TRAIN_RAW_DIR= "c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\raw\\train"
EVAL_RAW_DIR="c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\raw\\eval"
TRAIN_TOKENS="c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\processed\\train_tokens.txt"
EVAL_TOKENS="c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\processed\\eval_tokens.txt"
MODEL="c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\model\\model.json"
VOCAB="c:\\AI\\python\\practice\\ngram_rep\\ngram-predictor\\data\\model\\vocab.json"
"""

UNK_THRESHOLD=3
TOP_K=3
NGRAM_ORDER=4
hany = Normalizer().load(TRAIN_RAW_DIR)
hany2 = Normalizer().strip_gutenberg(hany)
hany3 = Normalizer().normalizer(hany2)
hany4a = Normalizer().sentence_tokenize(hany3)
##hany4b = Normalizer().word_tokenize(hany4a)
hany5 = Normalizer().write_list_to_file(hany4a,'',TRAIN_TOKENS)
#print(hany4b)
