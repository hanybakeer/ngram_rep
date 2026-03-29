# add main() if __name__=="__main__"

import os
import re
import string


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
        # 1) lowercase
        text = text.lower()

        # 2) remove numbers
        text = re.sub(r'\d+', '', text)

        # 3) remove punctuation, but keep . ! ?
        punctuation_to_remove = string.punctuation.replace('.', '').replace('!', '').replace('?', '')
        text = text.translate(str.maketrans('', '', punctuation_to_remove))

        # 4)    remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # 5) remove spaces before sentence punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)

        return text

#tests
#hany = Normalizer().load(r"C:\AI\python\practice\ngram_rep\ngram-predictor\data\raw\train")
hany = Normalizer().load(r"D:\AI\Python\practice\ngram_capstone\ngram_rep\ngram-predictor\data\raw\train\test")
hany2 = Normalizer().strip_gutenberg(hany)
hany3 = Normalizer().normalizer(hany2)

print(hany3)


