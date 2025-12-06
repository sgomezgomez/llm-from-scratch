import urllib.request, json, re, os
from config.config import data_config, tokenizer_config
from collections import Counter
from typing import Any, List, Tuple, Optional

# Simple BPE implementation from scratch
# Adapted from:https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb
class BPETokenizerSimple:
    def __init__(self, load_vocab: bool = False):
        if load_vocab:
            self._load_vocab_and_merges()
        else:
            self.vocab = {} # Token id to token string
        self.inv_vocab = {} # Token string to token id
        self.bpe_merges = {} # Dictionary of bpe merges
        self.bpe_ranks = {} # Dictionary of bpe ranks -- lower rank, higher priority
    
    def train(self, text: str, vocab_size: int, allowed_special: Optional[set[str]] = {"<|endoftext|>"}):
        """Train the BPE tokenizer from scratch"""
        # Preprocess: Replace spaces with "Ġ" where "Ġ" is the byte order mark for a space
        # Raschkas implementation uses a for loop to create a list
        preprocessed_text = text.replace(" ", "Ġ")

        # Initialize vocab with unique characters, including "Ġ" if present
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend([char for char in preprocessed_text if char not in unique_chars])
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")
        self.vocab = {id: char for id, char in enumerate(unique_chars)}
        self.inv_vocab = {char: id for id, char in self.vocab.items()}

        # Especial tokens
        for special in allowed_special:
            if special not in self.inv_vocab:
                id = len(self.vocab)
                self.vocab[id] = special
                self.inv_vocab[special] = id
        
        # Tokenize preprocessed text
        token_ids = [self.inv_vocab[char] for char in preprocessed_text]

        # BPE steps
        for _ in range(len(self.vocab), vocab_size):
            # Find frequent pair
            pair = self.get_most_frequent_pair(token_ids)
            if pair is None:
                break
            token_ids = self.merge_pair(token_ids, pair, _)
            merged_token = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.bpe_merges[pair] = _
            self.bpe_ranks[pair] = _ - len(self.vocab)  # Rank starts from 0
            self.vocab[_] = merged_token
            self.inv_vocab[merged_token] = _
        
        self._save_vocab_and_merges()
    
    def encode(self, text: str, allowed_special: Optional[set[str]] = {"<|endoftext|>"}, use_gpt2: bool = False) -> List[int]:
        """Encode a text into a list of integers including tiktoken style handling of special tokens"""
        self._load_vocab_and_merges(use_gpt2=use_gpt2)
        token_ids: List[int] = []
        
        # Preprocess text for GPT-2 compatibility
        if use_gpt2:
            # Convert tabs to spaces, then replace spaces with "Ġ" (GPT-2's space representation)
            text = text.replace("\t", "    ")  # Convert tabs to 4 spaces
            text = text.replace(" ", "Ġ")      # Replace spaces with Ġ
            
        disallowed = [token for token in self.inv_vocab if token.startswith("<|") and token.endswith("|>") and token not in allowed_special]
        if disallowed:
            raise ValueError(f"Disallowed special tokens found in vocabulary: {disallowed}")

        # If allowed special tokens
        if allowed_special is not None and len(allowed_special) > 0:
            # Regular expression to match allowed special tokens
            pattern = ("(" + "|".join(re.escape(token) for token in sorted(allowed_special, key=len, reverse=True)) + ")")

            last_index = 0
            for match in re.finditer(pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self._encode_plain(prefix)) # Encode prefix without special handling
                special_token = match.group(0)
                if special_token in self.inv_vocab:
                    token_ids.append(self.inv_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
            
            # Remaining tail after last special
            text = text[last_index:]
        
        token_ids.extend(self._encode_plain(text))
        return token_ids
    
    def _encode_plain(self, text: str) -> List[int]: # No special tokens
        """Encode a text into a list of integers without special tokens"""
        token_ids = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Split by whitespace but keep whitespace as separate elements
            parts = re.split(r'(\s+)', line)
            
            # Tokenize each part
            for part in parts:
                if part.isspace():  # Whitespace character(s)
                    # Convert spaces to "Ġ", keep other whitespace as-is
                    part = "".join("Ġ" if c == " " else c for c in part)
                
                if part in self.inv_vocab:
                    token_ids.append(self.inv_vocab[part])
                else:
                    token_ids.extend(self._tokenize_part(part))
            
            # Add newline after each line except the last one (if text doesn't end with newline)
            if i < len(lines) - 1:
                token_ids.append(self.inv_vocab["\n"])
        
        return token_ids

    def _tokenize_part(self, word: str) -> List[int]:
        """Tokenize part using BPE merges with rank-based priority"""        
        # Check if all characters are in vocabulary
        symbols = list(word)
        missing_chars = [char for char in symbols if char not in self.inv_vocab]
        if missing_chars:
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # Apply merges like training: find best pair, merge all occurrences, repeat
        while len(symbols) > 1:
            # Find the best pair (lowest rank) among current adjacent pairs
            best_rank = float("inf")
            best_pair = None
            
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            
            # If no valid pair found, we're done
            if best_pair is None:
                break
            
            # Merge all occurrences of the best pair
            first, second = best_pair
            merged_symbol = first + second
            
            # Helper function for readability
            def get_symbol(i):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    return merged_symbol
                elif i > 0 and symbols[i - 1] == first and symbols[i] == second:
                    return None  # Skip this element (already merged)
                else:
                    return symbols[i]
            
            # Create new list and filter out None values
            symbols = [s for s in [get_symbol(i) for i in range(len(symbols))] if s is not None]

        # Finally, convert merged symbols back to IDs
        token_ids = [self.inv_vocab[sym] for sym in symbols]
        return token_ids
    
    def decode(self, token_ids: List[int], use_gpt2: bool = False) -> str:
        """Decode a list of integers into a text"""
        self._load_vocab_and_merges(use_gpt2=use_gpt2)
        text = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token id {token_id} not found in vocabulary")
            text += self.vocab[token_id].replace("Ġ", " ")
        return text
    
    @staticmethod
    def get_most_frequent_pair(token_ids: List[int]) -> Tuple[int, int]:
        """Find the most frequent pair of tokens"""
        pair_counts = Counter(zip(token_ids, token_ids[1:]))
        if not pair_counts:
            return None
        return pair_counts.most_common(1)[0][0]
        
    @staticmethod
    def merge_pair(token_ids: List[int], pair: Tuple[int, int], new_token_id: int) -> List[int]:
        """Merge a pair of tokens"""
        new_token_ids = []
        i = 0
        # Loop until second-to-last element
        while i < (len(token_ids) - 1):
            if token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
                new_token_ids.append(new_token_id)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        
        # Handle last element if it exists
        if i < len(token_ids):
            new_token_ids.append(token_ids[i])
        
        return new_token_ids

    def _load_vocab(self, file_path: str) -> tuple:
        """Load vocabulary from JSON file {token_string: token_id}"""
        with open(file_path, "r", encoding="utf-8") as f:
            loaded_vocab = json.load(f)
        
        # GPT-2 format: {token_string: token_id}
        vocab = {int(v): k for k, v in loaded_vocab.items()}
        inv_vocab = {k: int(v) for k, v in loaded_vocab.items()}
        
        # Handle newline character without adding a new token
        if "\n" not in inv_vocab:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next((token for token in ["<|endoftext|>", "Ġ", ""] if token in inv_vocab), None)
            if fallback_token is not None:
                newline_token_id = inv_vocab[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")
            
            inv_vocab["\n"] = newline_token_id
            vocab[newline_token_id] = "\n"
        
        return vocab, inv_vocab

    def _load_merges(self, file_path: str) -> dict:
        """Load merges from text file"""
        merges = {}
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Skip version line
        start_line = 1 if lines and lines[0].startswith("#version:") else 0
        
        for i, line in enumerate(lines[start_line:], start_line):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    new_token_id = len(self.vocab) + len(merges)
                    merged_token = parts[0] + parts[1]
                    self.vocab[new_token_id] = merged_token
                    merges[(parts[0], parts[1])] = new_token_id
        
        return merges

    def _save_vocab_and_merges(self):
        """Save the vocabulary and merges to custom files"""
        vocab_path = data_config.get_data_path(tokenizer_config.custom_vocab_filename)
        merges_path = data_config.get_data_path(tokenizer_config.custom_merges_filename)
        
        # Save vocabulary in GPT-2 format: {token_string: token_id}
        vocab = {v: k for k, v in self.vocab.items()}
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Save BPE merges in text format (same as GPT-2)
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair, new_id in self.bpe_merges.items():
                f.write(f"{pair[0]} {pair[1]}\n")
    
    def _load_vocab_and_merges(self, use_gpt2: bool = False):
        """Load the vocabulary and merges from files"""
        # Check if already loaded
        if use_gpt2 == False and self.vocab and self.inv_vocab and self.bpe_merges and self.bpe_ranks:
            return  # Already loaded, no need to read files
        
        if use_gpt2:
            # Use gpt2 filenames
            vocab_filename = tokenizer_config.gpt2_vocab_filename
            merges_filename = tokenizer_config.gpt2_merges_filename
            vocab_path = data_config.get_data_path(vocab_filename)
            merges_path = data_config.get_data_path(merges_filename)
            if not os.path.exists(vocab_path):
                urllib.request.urlretrieve(tokenizer_config.gpt2_encoder_json_url, vocab_path)
                print(f"Downloading Tiktoken vocabulary from {tokenizer_config.gpt2_encoder_json_url} to {vocab_path}")
            if not os.path.exists(merges_path):
                urllib.request.urlretrieve(tokenizer_config.gpt2_vocab_bpe_url, merges_path)
                print(f"Downloading Tiktoken encoder from {tokenizer_config.gpt2_vocab_bpe_url} to {merges_path}")
        
        else:
            # Use custom filenames
            vocab_filename = tokenizer_config.custom_vocab_filename
            merges_filename = tokenizer_config.custom_merges_filename
            vocab_path = data_config.get_data_path(vocab_filename)
            merges_path = data_config.get_data_path(merges_filename)
            if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
                raise FileNotFoundError(f"Custom vocabulary files not found: {vocab_path}, {merges_path}")
        # Load vocabulary and merges
        self.vocab, self.inv_vocab = self._load_vocab(vocab_path)
        self.bpe_merges = self._load_merges(merges_path)
        
        # Rebuild ranks from merges (works for both custom and tiktoken files)
        self.bpe_ranks = {
            tuple(pair_str.split()) if isinstance(pair_str, str) else pair_str: rank
            for rank, (pair_str, new_token_id) in enumerate(self.bpe_merges.items())
        }

if __name__ == "__main__":
    # Sample text
    # Test encoding and decoding
    sample_text = """Hello world! This is a test.
With newlines and special chars: @#$%^&*()
And some code: def test(): return "hello"
Tabs:	here	too!
More    spaces    here    and    there
Mixed		tabs	and		spaces		everywhere
Python code:
    def hello():
        print("hi")
        return "done"
    
    class Test:
        def __init__(self):
            self.value = 42"""
    print(f"\nOriginal sampletext:\n{sample_text}")
    
    # Test 1: Train from scratch and encode/decode
    print("=== Testing Custom Training ===")
    
    # Load training text using SimpleTokenizer's method
    from tokenizer import SimpleTokenizer
    simple_tokenizer = SimpleTokenizer()
    training_text = simple_tokenizer._load_training_text()
    #print(f"Loaded training text ({len(training_text)} characters)")
    
    # Initialize and train tokenizer
    tokenizer = BPETokenizerSimple()
    print("Training tokenizer...")
    # Use a larger vocab size to allow for BPE merges
    tokenizer.train(
        text=training_text,
        vocab_size=5000,
        allowed_special={"<|endoftext|>"}
    )
    print(f"Vocab size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.bpe_merges)}")
    
    # Encoding sample text    
    encoded = tokenizer.encode(
        text=sample_text,
        use_gpt2=False
    )
    # Decoding sample text
    decoded = tokenizer.decode(
        token_ids=encoded,
        use_gpt2=False
    )
    print(f"Round-trip successful: {sample_text == decoded}")
    
    # Test 2: Load tiktoken data and encode/decode
    print("\n=== Testing Tiktoken Loading ===")
    
    # Create a new tokenizer instance
    tiktoken_tokenizer = BPETokenizerSimple()

    # This will download tiktoken files if they don't exist
    print("Loading tiktoken vocabulary and merges...")
    tiktoken_tokenizer._load_vocab_and_merges(use_gpt2=True)
    print(f"Tiktoken vocab size: {len(tiktoken_tokenizer.vocab)}")
    print(f"Tiktoken merges: {len(tiktoken_tokenizer.bpe_merges)}")
    
    encoded2 = tiktoken_tokenizer.encode(
        text=sample_text,
        use_gpt2=True
    )
    
    decoded2 = tiktoken_tokenizer.decode(
        token_ids=encoded2,
        use_gpt2=True
    )
    # For GPT-2, we need to account for tab-to-space conversion
    # Convert tabs to spaces in original for fair comparison (same as preprocessing)
    sample_text_normalized = sample_text.replace("\t", "    ")
    print(f"Round-trip successful: {sample_text_normalized == decoded2}")
    
    print("\n=== Testing Complete ===")