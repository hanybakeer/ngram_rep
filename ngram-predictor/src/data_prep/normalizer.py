from ast import Load


class Normalizer:
    "This class describes the module's responsibility: loading, cleaning, tokenizing, and saving the corpus."
    
    def load(self, file_path: str) -> str:

        """
        Load the raw corpus from a file.
        Parameters:
            file_path (str): Path to the input corpus file.
        Returns:
            str: The raw text loaded from the file.
        """

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    def clean(self, text: str) -> str:

        """
        Clean the raw text by removing unwanted characters and normalizing whitespace.
        Parameters:
            text (str): The raw text to be cleaned.
        """
        # Remove unwanted characters (e.g., punctuation, special characters)
        cleaned_text = ''.join(char for char in text if char.isalnum() or char.isspace())
        
        # Normalize whitespace (e.g., convert multiple spaces to a single space)
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text
    
    def tokenize(self, text: str) -> list:

        """
        Tokenize the cleaned text into a list of words.
        Parameters:
            text (str): The cleaned text to be tokenized.
        Returns:
            list: A list of tokens (words) extracted from the cleaned text.
        """
        return text.split() 
    
    def save(self, tokens: list, output_path: str) -> None:

        """
        Save the list of tokens to a file.
        Parameters:
            tokens (list): The list of tokens to be saved.
            output_path (str): Path to the output file where tokens will be saved.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(' '.join(tokens))
    
    
          