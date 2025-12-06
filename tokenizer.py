import urllib.request, os, re
from config.config import data_config, tokenizer_config

class SimpleTokenizer:
    def __init__(self):
        self.unk_token = "<|unk|>"
        self.end_of_text_token = "<|endoftext|>"
        self.str_to_idx = self._create_vocabulary_from_sample_text()
        self.idx_to_str = {idx: word for word, idx in self.str_to_idx.items()}

    def _ensure_training_text_file(self):
        """Download tokenizer sample training text if not present in data directory - Explaining the concept of tokenization"""
        # Get full path to the file
        file_path = data_config.get_data_path(tokenizer_config.tokenizer_training_text_filename)
        url = tokenizer_config.tokenizer_training_text_url
        
        # Download if file doesn't exist
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloading tokenizer training text from {url} to {file_path}")
        
        # Store file path
        self.training_text_file_path = file_path
    
    def _load_training_text(self):
        """Load tokenizer sample training text from the file path - Explaining the concept of tokenization"""
        self._ensure_training_text_file()
        with open(self.training_text_file_path, "r", encoding="utf-8") as file:
            return file.read()
    
    def _split_text(self, text):
        """Split text into words - Explaining the concept of tokenization"""
        preprocessed_text = re.split(r"([,.:;?_!\"()']|--|\s)", text)
        preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
        return preprocessed_text
    
    def _create_vocabulary_from_sample_text(self):
        """Create a vocabulary from the training text - Explaining the concept of tokenization"""
        preprocessed_text = self._split_text(self._load_training_text())
        unique_words = sorted(set(preprocessed_text))
        # Extend vocabulary with special tokens
        unique_words.extend([self.end_of_text_token, self.unk_token])
        return {word: idx for idx, word in enumerate(unique_words)}
    
    def encode(self, text):
        """Encode a text into a list of integers - Explaining the concept of tokenization"""
        preprocessed_text = self._split_text(text)
        preprocessed_text = [item if item in self.str_to_idx else self.unk_token for item in preprocessed_text]
        return [self.str_to_idx[word] for word in preprocessed_text if word in self.str_to_idx]
    
    def decode(self, ids):
        """Decode a list of integers into a text - Explaining the concept of tokenization"""
        text = " ".join([self.idx_to_str[idx] for idx in ids])
        return re.sub(r" ([,.:;?_!\"()']|--)", r"\1", text)

class TiktokenTokenizer:
    def __init__(self):
        import tiktoken
        self.tiktoken_ = tiktoken.get_encoding("gpt2")
    
    def encode(self, text):
        """Use tiktoken to tokenize a text - Explaining the concept of tokenization"""
        return self.tiktoken_.encode(text, allowed_special="all")
    
    def decode(self, ids):
        """Use tiktoken to decode a list of integers into a text - Explaining the concept of tokenization"""
        return self.tiktoken_.decode(ids)


if __name__ == "__main__":
    tokenizer = SimpleTokenizer()
    tiktoken_tokenizer = TiktokenTokenizer()
    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Text: {text}")
    print(f"SimpleTokenizer Ids: {ids}")
    print(f"SimpleTokenizer Decoded: {decoded}")
    ids = tiktoken_tokenizer.encode(text)
    decoded = tiktoken_tokenizer.decode(ids)
    print(f"tiktoken Ids: {ids}")
    print(f"tiktoken Decoded: {decoded}")