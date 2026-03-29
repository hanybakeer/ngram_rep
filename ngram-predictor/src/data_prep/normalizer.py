# add main() if __name__=="__main__"

# from ast import Load
#from email.mime import text
import os
import re
#from pickle import load
#from tracemalloc import start


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
        
        #text = ""
        #header_pattern = "*** START OF THE PROJECT GUTENBERG EBOOK"
        #footer_pattern = "*** END OF THE PROJECT GUTENBERG EBOOK"
        #header_index = in_text.find(header_pattern)
        #footer_index = in_text.find(footer_pattern)
        #print(header_index) 
        #print(footer_index)
        #part = in_text[header_index:footer_index]
        #part = in_text[3:4]


#tests
#hany = Normalizer().load(r"C:\AI\python\practice\ngram_rep\ngram-predictor\data\raw\train")
hany = Normalizer().load(r"D:\AI\Python\practice\ngram_capstone\ngram_rep\ngram-predictor\data\raw\train")
hany2 = Normalizer().strip_gutenberg(hany)

print(hany2)


