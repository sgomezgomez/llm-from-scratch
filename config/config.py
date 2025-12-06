import os
from typing import Optional
from dotenv import load_dotenv

class Config:
    """Base configuration class."""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Get environment variable with optional default value."""
        return os.getenv(key, default)

class DataConfig(Config):
    """Configuration class for data-related settings."""
    
    def __init__(self):
        super().__init__()
        # Read and store the environment variable once during initialization
        self.data_dir = self.get_env_var("DATA_DIR")
        self.plot_dir = self.data_dir + "/" + self.get_env_var("PLOT_DIR")
        self.model_dir = self.data_dir + "/" + self.get_env_var("MODEL_DIR")
        # Pretraining related directories
        self.pretraining_dir = self.data_dir + "/" + self.get_env_var("PRETRAINING_DIR")
        # Single cache directory for both index and tokenized files (organized in subdirectories)
        self.pretraining_cache_dir = self.pretraining_dir + "/" + self.get_env_var("PRETRAINING_CACHE_DIR")
        # Pretraining data directories (comma-separated list in env var)
        pretraining_data_dirs_str = self.get_env_var("PRETRAINING_DATA_DIRS", "")
        self.pretraining_data_dirs = [os.path.join(self.pretraining_dir, d.strip()) for d in pretraining_data_dirs_str.split(",") if d.strip()]
    
    def get_data_path(self, filename: str) -> str:
        """Get full path to a file in the data directory."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        return os.path.join(self.data_dir, filename)

class TokenizerConfig(Config):
    def __init__(self):
        super().__init__()
        self.tokenizer_training_text_filename = self.get_env_var("TOKENIZER_TRAINING_TEXT_FILENAME")
        self.tokenizer_training_text_url = self.get_env_var("TOKENIZER_TRAINING_TEXT_URL")
        # URLs for downloading GPT-2 tokenizer files
        self.gpt2_encoder_json_url = self.get_env_var("GPT2_ENCODER_JSON")
        self.gpt2_vocab_bpe_url = self.get_env_var("GPT2_VOCAB_BPE")
        self.gpt2_vocab_filename = self.get_env_var("GPT2_VOCAB_FILENAME")
        self.gpt2_merges_filename = self.get_env_var("GPT2_MERGES_FILENAME")
        self.custom_vocab_filename = self.get_env_var("CUSTOM_VOCAB_FILENAME")
        self.custom_merges_filename = self.get_env_var("CUSTOM_MERGES_FILENAME")

# Global instance variable (initially None)
_data_config = None
_tokenizer_config = None

def get_data_config() -> DataConfig:
    """Get the global data config instance, creating it if it doesn't exist."""
    global _data_config
    if _data_config is None:
        _data_config = DataConfig()
    return _data_config

def get_tokenizer_config() -> TokenizerConfig:
    """Get the global tokenizer config instance, creating it if it doesn't exist."""
    global _tokenizer_config
    if _tokenizer_config is None:
        _tokenizer_config = TokenizerConfig()
    return _tokenizer_config

# Create a global instance for easy access
data_config = get_data_config()
tokenizer_config = get_tokenizer_config()