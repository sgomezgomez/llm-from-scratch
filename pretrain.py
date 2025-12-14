# Adapted from: 
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/03_bonus_pretraining_on_gutenberg/pretraining_simple.py

# Standard library imports
import argparse, datetime, hashlib, json, math, multiprocessing, os, platform, subprocess, time, random, csv
from pathlib import Path
from typing import Literal, NoReturn
# Third-party imports
import matplotlib.pyplot as plt, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, tiktoken
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
# Custom imports
from config.config import get_data_config
from utils.paths import normalize_path, is_wsl
from gpt import GPTModel

def configure_dataloader_workers(user_specified: int | None = None, verbose: bool = True) -> int:
    """Configure DataLoader workers with multiprocessing setup and auto-detection."""
    system = platform.system()
    
    # Determine number of workers
    if user_specified is not None:
        num_workers = user_specified
    else:
        cpu_count = os.cpu_count() or 1
        #num_workers = min(2, max(1, cpu_count // 4)) if system == 'Windows' else min(4, max(2, cpu_count // 2))
        if system == 'Windows':
            # On Windows, spawn is slower but still beneficial for I/O-bound tasks
            # Use 2-4 workers depending on CPU count
            #num_workers = min(2, max(2, cpu_count // 4)) if cpu_count else 2
            num_workers = 0
        else:
            # Linux/macOS: fork is fast, use more workers
            #num_workers = min(4, max(4, cpu_count // 2))
            num_workers = min(0, max(4, cpu_count // 2))
    
    # Configure multiprocessing for Windows if using workers
    if system == 'Windows' and num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    if verbose:
        print("\n" + "="*60)
        print("DataLoader Worker Configuration")
        print("="*60)
        if user_specified is not None:
            print(f"Using user-specified worker count: {num_workers}")
        else:
            cpu_count = os.cpu_count() or 1
            if system == 'Windows':
                print(f"Windows detected: Auto-selecting {num_workers} workers (spawn method)")
                print(f"  CPU cores available: {cpu_count}\n  Calculation: min(2, max(1, {cpu_count} // 4)) = {num_workers}")
                print(f"  Note: Spawn is slower than fork, but may still help with I/O-bound tasks")
            else:
                print(f"{system} detected: Auto-selecting {num_workers} workers")
                print(f"  CPU cores available: {cpu_count}\n  Calculation: min(4, max(2, {cpu_count} // 2)) = {num_workers}")
        print(f"\nFinal decision: Using {num_workers} DataLoader worker(s)")
        print("="*60 + "\n")
    
    return num_workers

def decode_with_fallback(content_bytes: bytes, fallback_encoding: str = "latin1") -> str:
    """Decode bytes with UTF-8, fallback to specified encoding if needed."""
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode(fallback_encoding)

def read_file(file_path: str,
              start_byte: int = None, # If None, read from start of file
              bytes_to_read: int = None, # If None, read entire file
              fallback_encoding: str = "latin1") -> str: # UTF-8 fallback
    """Read file from start_byte to bytes_to_read with encoding fallback."""
    assert os.path.exists(file_path), f"File {file_path} does not exist"
    assert (start_byte is not None and bytes_to_read is not None) or (start_byte is None and bytes_to_read is None), "start_byte and bytes_to_read must be provided together or both be None"
    
    # Read in binary mode, then decode with fallback
    with open(file_path, "rb") as file:
        if start_byte is not None:
            file.seek(start_byte)
        if bytes_to_read is not None:
            content_bytes = file.read(bytes_to_read)
        else:
            content_bytes = file.read()
    return decode_with_fallback(content_bytes, fallback_encoding)

def discover_files(data_dirs: list[str], file_extensions: tuple[str, ...] = (".txt",)) -> list[str]:
        """Discover all files with given extensions in specified directories."""
        all_files = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: Data directory does not exist: {data_dir}")
                continue
            for path, subdirs, files in os.walk(data_dir):
                for name in files:
                    if name.endswith(file_extensions):
                        all_files.append(os.path.join(path, name))
        return sorted(all_files)  # Sort for reproducibility

def get_gpu_stats(device: torch.device):
    """Get GPU memory statistics"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        utilization_pct = (allocated / total_memory) * 100 if total_memory > 0 else 0
        print(f"{'='*60}")
        print(
            f"\n{'='*60}\n"
            f"  GPU Stats:\n"
            f"{'='*60}\n"
            f"    Device:            {device}\n"
            f"    GPU allocated:     {allocated:.2f} GB ({utilization_pct:.1f}%)\n"
            f"    GPU reserved:      {reserved:.2f} GB\n"
            f"    GPU peak:          {max_allocated:.2f} GB\n"
            f"{'='*60}\n"
        )
        print(f"{'='*60}")

def calculate_batch_loss(model, 
                         input_batch: torch.Tensor, target_batch: torch.Tensor, 
                         device: torch.device, pad_token: int) -> torch.Tensor:
    input_batch = input_batch.to(device, non_blocking=True)  # Non-blocking transfer for better GPU utilization
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(input_batch)
    
    # Mask to ignore padded positions (don't compute loss for padding tokens)
    mask = (target_batch != pad_token)
    
    # Flatten logits and targets
    logits_flat = logits.flatten(0, 1)  # [batch * seq_len, vocab_size]
    targets_flat = target_batch.flatten()  # [batch * seq_len]
    mask_flat = mask.flatten()  # [batch * seq_len]
    
    # Compute loss for non-padded positions
    # If all positions are padded, return 0 loss
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=device)
    loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
    return loss

def calculate_loss(model, dataloader: DataLoader, 
                   device: torch.device, pad_token: int = None,
                   num_batches: int = None,
                   verbose: bool = False) -> float:
    """Calculate average loss over batches."""
    assert num_batches > 0 if num_batches is not None else True, "num_batches must be non-negative if provided"    
    
    # Sampling
    if num_batches is not None:
        # Use RandomSampler with DataLoader for efficient random sampling
        # This leverages DataLoader optimizations (prefetching, multiprocessing)
        assert hasattr(dataloader, 'sample_dataloader'), "DataLoader must have a sample_dataloader attribute"
        eval_loader = dataloader.sample_dataloader
        
        total_loss = 0.0
        batches_processed = 0
        data_retrieval_times, step_times = [], []
        data_start = time.perf_counter()

        if verbose:
            print(f"Evaluation: {dataloader.dataset.mode} mode.")

        for inputs, targets in eval_loader:
            # Time batch retrieval
            retrieval_time = time.perf_counter() - data_start
            data_retrieval_times.append(retrieval_time)
            
            step_start_time = time.perf_counter()
            loss = calculate_batch_loss(model, inputs, targets, device, pad_token)
            step_duration = time.perf_counter() - step_start_time
            step_times.append(step_duration)
            
            if verbose:
                print(f"    Batch {batches_processed + 1}/{num_batches}")
                print(f"    Retrieval time: {retrieval_time:.3f}s")
                print(f"    Forward pass time: {step_duration:.3f}s")
                print(f"    Total time: {retrieval_time + step_duration:.3f}s")
            
            total_loss += loss.item()
            batches_processed += 1
            data_start = time.perf_counter()
        
        avg_data_time = sum(data_retrieval_times) / len(data_retrieval_times)
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
        if verbose:
            print(f"  Average retrieval time: {avg_data_time:.3f}s (min: {min(data_retrieval_times):.3f}s, max: {max(data_retrieval_times):.3f}s)")
            print(f"  Average forward pass time: {avg_step_time:.3f}s (min: {min(step_times):.3f}s, max: {max(step_times):.3f}s)")
        
        # Average over actual number of batches processed
        if batches_processed == 0:
            return float("nan")
        return total_loss / batches_processed
    else:
        # Evaluate on all batches sequentially
        # No verbose mode
        total_loss = 0.0
        batches_processed = 0
        for inputs, targets in dataloader:
            loss = calculate_batch_loss(model, inputs, targets, device, pad_token)
            total_loss += loss.item()
            batches_processed += 1
        
        if batches_processed == 0:
            return float("nan")
        return total_loss / batches_processed

def evaluate_model(model, train_loader: DataLoader, val_loader: DataLoader, 
                   device: torch.device, pad_token: int = None,
                   train_eval_batches: int = 5, val_eval_batches: int = 5,
                   verbose: bool = False) -> tuple[float, float]:
    """Evaluate model on training (sampled) and validation (sampled) datasets."""
    
    if verbose:
        print(f"{'='*60}")
        print(f"Evaluating model on training and validation datasets...")
    
    # Set model to evaluation mode
    model.eval() # Dropout is disabled during evaluation for stability
    # No gradient tracking to reduce computation overhead
    with torch.no_grad():
        # Training: sample random batches
        train_loss = calculate_loss(
            model=model,
            dataloader=train_loader,
            device=device,
            pad_token=pad_token,
            num_batches=train_eval_batches,
            verbose=verbose
        )
        # Validation: sample random batches
        val_loss = calculate_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            pad_token=pad_token,
            num_batches=val_eval_batches,
            verbose=verbose
        )
    model.train() # Set model back to training mode
    if verbose:
        print(f"{'='*60}")
    return train_loss, val_loss

def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)

def format_tokens(num_tokens: int) -> str:
    """Format token count in a human-readable way (e.g., 1.2M, 500K)."""
    if num_tokens >= 1_000_000:
        return f"{num_tokens / 1_000_000:.2f}M"
    elif num_tokens >= 1_000:
        return f"{num_tokens / 1_000:.1f}K"
    else:
        return str(num_tokens)

def print_token_statistics(train_loader, val_loader):
    """Print token statistics for training and validation datasets."""
    # Calculate total training and validation tokens
    # Training tokens: sequences * batch_size * max_length (approximate, actual may vary due to padding)
    total_train_sequences = len(train_loader.dataset)
    total_train_tokens = total_train_sequences * train_loader.batch_size * train_loader.dataset.max_length
    # Validation tokens: sequences * batch_size * max_length (approximate)
    total_val_sequences = len(val_loader.dataset)
    total_val_tokens = total_val_sequences * val_loader.batch_size * val_loader.dataset.max_length
    
    # Print token statistics
    print(f"\n{'='*60}")
    print("Dataset Token Statistics")
    print("="*60)
    print(f"Training dataset:")
    print(f"   Sequences: {total_train_sequences:,}")
    print(f"   Estimated tokens: {format_tokens(total_train_tokens)} ({total_train_tokens:,})")
    print(f"Validation dataset:")
    print(f"   Sequences: {total_val_sequences:,}")
    print(f"   Estimated tokens: {format_tokens(total_val_tokens)} ({total_val_tokens:,})")
    print(f"Total estimated tokens: {format_tokens(total_train_tokens + total_val_tokens)} ({total_train_tokens + total_val_tokens:,})")
    print(f"{'='*60}\n")

def save_checkpoint(model, checkpoint_dir: str, global_step: int = None, verbose: bool = False):
    if checkpoint_dir is None:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if global_step is not None:
        checkpoint_filename = f"checkpoint_step_{global_step:07d}_{timestamp}.pth"
        message = f"\nCheckpoint saved at step {global_step:,}: "
    else:
        checkpoint_filename = f"{timestamp}.pth"
        message = "Model saved to "
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    model.save_model(checkpoint_path)
    
    if verbose:
        print(f"{message}{checkpoint_path}")

def generate_sample(model, tokenizer, 
                    device: torch.device, 
                    max_tokens: int):
    assert hasattr(model, 'generate') and callable(getattr(model, 'generate')), \
        "Model must have a 'generate' method"
    
    prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
    ]
    prompt = random.choice(prompts)
    
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt, allowed_special="all")], device=device)
    
    # Collect all tokens from generator
    # Assumes generate uses streaming by default
    generated_tokens = []
    for token in model.generate(
        idx, 
        max_tokens=max_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.9
    ):
        generated_tokens.append(token)
    
    # Concatenate initial context with generated tokens
    all_tokens = torch.cat([idx] + generated_tokens, dim=1) if generated_tokens else idx
    text = tokenizer.decode(all_tokens.squeeze(0).tolist())
    print(
        f"\n{'='*60}\n"
        f"  Generated text:\n"
        f"{'='*60}\n"
        f"{text.replace('\n', ' ')}\n"
        f"{'='*60}\n"
    )
    model.train()

def plot_loss(global_step: int, step_losses: list[float],
              eval_steps: list[int], eval_tokens: list[int],
              eval_train_losses: list[float], eval_val_losses: list[float],
              save_path: str) -> None:
    """Plot per-step and per-eval losses with tokens on twin x-axis."""
    fig, ax_loss = plt.subplots(figsize=(8, 5))

    # Per-step training loss (primary x-axis: global step)
    ax_loss.plot(range(1, len(step_losses) + 1), step_losses,
                 color='tab:blue', alpha=0.5, label='Train Loss (step)')

    # Per-eval losses (overlaid markers using actual eval steps)
    # Train loss
    ax_loss.plot(eval_steps, eval_train_losses, marker='o', linestyle='-',
                 color='tab:green', label='Train Loss (eval)')
    # Val loss
    ax_loss.plot(eval_steps, eval_val_losses, marker='s', linestyle='-.',
                 color='tab:red', label='Val Loss (eval)')

    ax_loss.set(xlabel='Global Steps', ylabel='Loss')
    ax_loss.legend(loc='upper right')
    ax_loss.grid(True, alpha=0.2)

    # Twin x-axis for tokens seen, aligned with steps
    ax_tokens = ax_loss.twiny()
    ax_tokens.set_xlim(ax_loss.get_xlim())
    ticks = [tick for tick in ax_loss.get_xticks() if tick >= 1]
    tick_indices = [min(int(tick) - 1, len(eval_tokens) - 1) for tick in ticks]
    tick_labels = [format_tokens(eval_tokens[idx]) for idx in tick_indices]
    ax_tokens.set_xticks(ticks)
    ax_tokens.set_xticklabels(tick_labels)
    ax_tokens.set_xlabel('Tokens Seen')

    # Save plot
    fig.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plotname = f"{save_path}/loss_step_{global_step}_{timestamp}.png"
    plt.savefig(plotname)

def append_csv_row(path: str, fieldnames: list[str], rows: list[dict]):
    """Append rows to CSV, writing header if file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_steps(steps_dir: str,
              epoch: int, global_step: int, tokens_seen: list[int],
              losses: list[float], retrieval_times: list[float], 
              compute_times: list[float], lr_values: list[float]) -> None:
    """Write per-step rows since last eval."""
    # Field names
    step_fields = [
        'timestamp', 'epoch', 'global_step', 'learning_rate',
        'step_loss', 'retrieval_time', 'compute_time'
    ]
    # Rows
    step_rows = []
    start_step = global_step - len(losses) + 1
    for idx in range(len(losses)):
        step_idx = start_step + idx
        step_rows.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'epoch': epoch,
            'global_step': step_idx,
            'learning_rate': lr_values[idx],
            'step_loss': losses[idx],
            'retrieval_time': retrieval_times[idx],
            'compute_time': compute_times[idx]
        })
    append_csv_row(os.path.join(steps_dir, 'step_metrics.csv'), step_fields, step_rows)

def log_eval(steps_dir: str, epoch: int,
             global_step: int, tokens_seen: int,
             train_loss: float, val_loss: float,
             lr_last: float, eval_time: float,
             total_elapsed_s: float, eta_hms: str) -> None:
    """Write one per-eval row."""
    # Per-eval row
    eval_fields = [
        'timestamp', 'epoch', 'global_step', 'tokens_seen',
        'train_loss', 'val_loss', 'lr_last',
        'eval_time', 'total_elapsed_s', 'eta_hms'
    ]

    eval_row = {
        'timestamp': datetime.datetime.now().isoformat(),
        'epoch': epoch,
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr_last': lr_last,
        'eval_time': eval_time,
        'total_elapsed_s': total_elapsed_s,
        'eta_hms': eta_hms
    }
    append_csv_row(os.path.join(steps_dir, 'eval_metrics.csv'), eval_fields, [eval_row])

class StreamingDataset(Dataset):
    """Dataset that reads files in chunks and tokenizes on-demand."""
    
    def __init__(self, tokenizer, # Should we validate tokenizer type?
                 max_length: int, stride: int, pad_token: int,
                 train_ratio: float = 0.95, mode: Literal['train', 'val'] = 'train',
                 force_build: bool = False) -> None:
        # Discover files
        data_dirs = get_data_config().pretraining_data_dirs
        self.file_paths = [normalize_path(f) for f in discover_files(data_dirs)]
        assert len(self.file_paths) > 0, f"No training files found in directories: {data_dirs}."
        
        # Save dataset config params
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.pad_token = pad_token
        self.train_ratio = train_ratio
        self.mode = mode
        self.force_build = force_build
        
        # Get tokenizer identifier for cache validation
        # For tiktoken, use the encoding name (e.g., "gpt2")
        if hasattr(tokenizer, 'name'):
            self.tokenizer_id = tokenizer.name
        elif hasattr(tokenizer, 'encoding_name'):
            self.tokenizer_id = tokenizer.encoding_name
        else:
            # Fallback: use string representation
            self.tokenizer_id = str(type(tokenizer).__name__)
        
        # Per-file train/val split indices
        self.file_sequences = []  # List of (file_path, num_sequences, train_split_idx) tuples
        self.sequences = []  # List of (file_idx, start_token_idx, end_token_idx) tuples for mode
        
        # Per-worker memory cache: LRU cache for multiple tokenized files
        # This helps when a worker accesses multiple sequences from the same file
        # Using OrderedDict for LRU eviction (max_cache_size files)
        self._max_cache_size = 100  # Number of files to cache per worker
        self._file_cache = OrderedDict()  # {file_idx: tokenized_numpy_array}
        self._file_mtimes = {}  # {file_idx: mtime} for cache validation
        self._cached_raw_paths = {}  # {normalized_path: original_cached_path}
        
        # Build index mapping with per-file splits (with caching)
        self._build_index_map()

    def _path_variants(self, path_str: str) -> list[str]:
        """Return path variants to allow reusing caches across Windows/WSL."""
        variants = [path_str]
        if path_str in self._cached_raw_paths:
            variants.append(self._cached_raw_paths[path_str])
        if is_wsl():
            try:
                win_path = subprocess.check_output(["wslpath", "-w", path_str], text=True).strip()
                variants.append(win_path)
            except Exception:
                pass
        # Deduplicate preserving order
        seen = set()
        uniq = []
        for p in variants:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq
    
    def _get_cache_key(self) -> str:
        config_str = json.dumps({
            "tokenizer": self.tokenizer_id,
            "max_length": self.max_length,
            "stride": self.stride,
            "pad_token": self.pad_token
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_cache_dir(self) -> Path:
        data_config = get_data_config()
        cache_dir = Path(data_config.pretraining_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _get_tokenized_cache_dir(self) -> Path:
        cache_dir = self._get_cache_dir() / "tokenized"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _get_index_cache_path(self) -> Path:
        cache_dir = self._get_cache_dir()
        cache_key = self._get_cache_key()
        return cache_dir / f"index_{cache_key}.json"
    
    def _get_tokenized_cache_path(self, file_path: str, file_mtime: float) -> Path:
        cache_dir = self._get_tokenized_cache_dir()
        # Create hash from file path + mtime + tokenizer to detect changes
        # Tokenizer is included because different tokenizers produce different tokenized content
        cache_key = hashlib.md5(f"{file_path}_{file_mtime}_{self.tokenizer_id}".encode()).hexdigest()
        # Use .npy for numpy
        return cache_dir / f"{cache_key}.npy"

    def _iter_tokenized_cache_paths(self, file_path: str, file_mtime: float):
        """Yield possible tokenized cache paths for different path variants (posix/windows)."""
        for variant in self._path_variants(file_path):
            yield self._get_tokenized_cache_path(variant, file_mtime)
    
    def _load_index_cache(self) -> dict:
        cache_path = self._get_index_cache_path()
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _save_index_cache(self, cache_data: dict):
        cache_path = self._get_index_cache_path()
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_tokenized_cache(self, file_path: str, file_mtime: float) -> np.ndarray | None:
        # Try multiple path variants (POSIX/Windows) to reuse caches built on different platforms
        for tokenized_cache_path in self._iter_tokenized_cache_paths(file_path, file_mtime):
            if tokenized_cache_path.exists():
                try:
                    arr = np.load(tokenized_cache_path, allow_pickle=False)
                    return arr
                except (IOError, ValueError, OSError):
                    continue
        return None
    
    def _save_tokenized_cache(self, file_path: str, file_mtime: float, tokenized: list[int] | np.ndarray):
        tokenized_cache_path = self._get_tokenized_cache_path(file_path, file_mtime)
        try:
            # Convert to numpy array if needed, use int32 for memory and disk
            if isinstance(tokenized, list):
                arr = np.array(tokenized, dtype=np.int32)
            else:
                arr = tokenized.astype(np.int32) if tokenized.dtype != np.int32 else tokenized
            np.save(tokenized_cache_path, arr, allow_pickle=False)
        except IOError:
            # Failed to write cache, continue without caching
            pass
    
    def _validate_cache_config(self, cached_config: dict) -> bool:
        current_config = {
            "tokenizer": self.tokenizer_id,
            "max_length": self.max_length,
            "stride": self.stride,
            "pad_token": self.pad_token
        }
        return cached_config == current_config
    
    def _build_sequences_for_file(self, file_idx: int, num_sequences: int, file_train_split_idx: int, tokenized_length: int):
        seq_range = range(file_train_split_idx) if self.mode == 'train' else range(file_train_split_idx, num_sequences)
        for seq_idx in seq_range:
            start_idx = seq_idx * self.stride
            end_idx = min(start_idx + self.max_length, tokenized_length)
            self.sequences.append((file_idx, start_idx, end_idx))
    
    def _build_index_map(self):
        """Builds index map per file with token indices based on mode (with caching)."""        
        # Check if cache file exists for this config (skip if force_build is True)
        index_cache_path = self._get_index_cache_path()
        cached_file_sequences = {}
        if not self.force_build and index_cache_path.exists():
            # Load index cache
            index_cache_data = self._load_index_cache()
            if index_cache_data:
                # Validate config matches (safety check - cache key should already ensure this)
                if not self._validate_cache_config(index_cache_data.get("config", {})):
                    raise ValueError(
                        f"Cache config mismatch! Cached: {index_cache_data.get('config')}, "
                        f"Current: {{'tokenizer': {self.tokenizer_id}, 'max_length': {self.max_length}, "
                        f"'stride': {self.stride}, 'pad_token': {self.pad_token}}}. "
                        f"Please delete cache, use matching config, or set force_build=True."
                    )
                
                # Build lookup dict from cached file_sequences
                for item in index_cache_data.get("file_sequences", []):
                    raw_path = item.get("file_path")
                    normalized_path = normalize_path(raw_path)
                    item["file_path"] = normalized_path
                    cached_file_sequences[normalized_path] = item
                    if raw_path:
                        self._cached_raw_paths[normalized_path] = raw_path
        
        # Track which files need to be processed
        files_to_process = []
        cached_count = 0
        pbar = tqdm(self.file_paths, desc=f"Building {self.mode} dataset index", leave=False)
        
        for file_path in pbar:
            file_path_str = normalize_path(str(file_path))
            file_mtime_current = os.path.getmtime(file_path_str) if os.path.exists(file_path_str) else None

            # Check if file is in cache and mtime matches
            if file_path_str in cached_file_sequences:
                cached_item = cached_file_sequences[file_path_str]
                cached_mtime = cached_item.get("file_mtime")
                
                if cached_mtime == file_mtime_current:
                    # Verify all required attributes are present
                    num_sequences = cached_item.get("num_sequences")
                    file_train_split_idx = cached_item.get("file_train_split_idx")
                    tokenized_length = cached_item.get("tokenized_length")

                    # If any required attribute is missing, process the file normally
                    if num_sequences is None or file_train_split_idx is None or tokenized_length is None:
                        files_to_process.append((file_path_str, file_mtime_current))
                        continue

                    # Prefer cached mtime for locating tokenized cache; fall back to current
                    mtime_candidates = [cached_mtime, file_mtime_current]
                    file_mtime_use = None
                    tokenized_exists = False
                    for mtime in mtime_candidates:
                        if mtime is None:
                            continue
                        if any(path.exists() for path in self._iter_tokenized_cache_paths(file_path_str, mtime)):
                            tokenized_exists = True
                            file_mtime_use = mtime
                            break

                    if not tokenized_exists:
                        # Tokenized cache file missing, need to recreate it
                        files_to_process.append((file_path_str, file_mtime_current))
                        continue

                    # Use cached data, no need to read/tokenize
                    file_idx = len(self.file_sequences)
                    self.file_sequences.append((file_path_str, num_sequences, file_train_split_idx))
                    # Store mtime used for this file to help downstream cache lookups
                    if file_mtime_use is not None:
                        self._file_mtimes[file_idx] = file_mtime_use

                    # Build sequences from cached metadata
                    self._build_sequences_for_file(file_idx, num_sequences, file_train_split_idx, tokenized_length)

                    # Load to cache if max_cache_size not reached
                    if len(self._file_cache) < self._max_cache_size:
                        tokenized = self._get_tokenized_file(file_idx)
                    
                    cached_count += 1
                    pbar.set_postfix({
                        'files': f'{len(self.file_sequences)}/{len(self.file_paths)}',
                        'cached': cached_count,
                        'sequences': len(self.sequences)
                    })
                    continue
            
            # File not in cache or mtime changed, need to process
            files_to_process.append((file_path_str, file_mtime))
        
        # Process new/changed files
        if files_to_process:
            process_pbar = tqdm(files_to_process, desc=f"Processing {len(files_to_process)} files", leave=False)
            for file_path_str, file_mtime in process_pbar:
                # Files in files_to_process didn't pass cache filters (not in cache, mtime changed, or incomplete)
                # So we need to read and tokenize fresh
                tokenized = self.tokenizer.encode(read_file(file_path_str), allowed_special="all")
                
                # Save to disk cache for future use
                # file_mtime should always be set (file exists, otherwise read_file would fail)
                self._save_tokenized_cache(file_path_str, file_mtime, tokenized)
                
                # Per-file split: calculate how many sequences go to train vs val
                num_sequences = max(1, (len(tokenized) - self.max_length) // self.stride + 1)
                file_train_split_idx = int(num_sequences * self.train_ratio)
                file_idx = len(self.file_sequences)
                self.file_sequences.append((file_path_str, num_sequences, file_train_split_idx))
                
                # Build sequences based on mode to avoid duplication
                self._build_sequences_for_file(file_idx, num_sequences, file_train_split_idx, len(tokenized))
                
                # Update cached_file_sequences for saving
                cached_file_sequences[file_path_str] = {
                    "file_path": file_path_str,
                    "num_sequences": num_sequences,
                    "file_train_split_idx": file_train_split_idx,
                    "file_mtime": file_mtime,
                    "tokenized_length": len(tokenized)
                }
                
                process_pbar.set_postfix({
                    'files': f'{len(self.file_sequences)}/{len(self.file_paths)}',
                    'sequences': len(self.sequences)
                })
        
        # Save updated cache if we processed new files
        if files_to_process:
            cache_data = {
                "config": {
                    "tokenizer": self.tokenizer_id,
                    "max_length": self.max_length,
                    "stride": self.stride,
                    "pad_token": self.pad_token
                },
                "file_sequences": list(cached_file_sequences.values())
            }
            self._save_index_cache(cache_data)
    
    def _get_tokenized_file(self, file_idx: int) -> np.ndarray:
        # Check if this file is already cached in this worker's memory (LRU cache)
        if file_idx in self._file_cache:
            # Move to end (most recently used)
            tokenized = self._file_cache.pop(file_idx)
            self._file_cache[file_idx] = tokenized
            return tokenized
        
        # Load file path and get mtime (cache this to avoid repeated stat calls)
        file_path, _, _ = self.file_sequences[file_idx]
        
        # Check if we have a cached mtime, otherwise get it
        if file_idx not in self._file_mtimes:
            self._file_mtimes[file_idx] = os.path.getmtime(file_path)
        file_mtime = self._file_mtimes[file_idx]
        
        # Try to load from disk cache first
        tokenized = self._load_tokenized_cache(file_path, file_mtime)
        
        if tokenized is None:
            # Tokenize file (cache miss or cache error)
            # tokenizer.encode returns a list, convert to numpy array
            tokenized_list = self.tokenizer.encode(read_file(file_path), allowed_special="all")
            tokenized = np.array(tokenized_list, dtype=np.int32)
            
            # Save to disk cache for future use
            self._save_tokenized_cache(file_path, file_mtime, tokenized)
        
        # Cache in this worker's memory using LRU eviction
        # Evict oldest entry if cache is full
        if len(self._file_cache) >= self._max_cache_size:
            evicted_file_idx, _ = self._file_cache.popitem(last=False)  # Remove oldest (first) item
            # Clean up mtime cache for evicted file to prevent memory leak
            if evicted_file_idx in self._file_mtimes:
                del self._file_mtimes[evicted_file_idx]
        
        self._file_cache[file_idx] = tokenized
        
        return tokenized
    
    def _get_sequence(self, tokenized: np.ndarray, start_idx: int, end_idx: int):
        # If at end of file (end_idx == len), exclude last token from input since we can't predict beyond it
        # This ensures input and target have the same length (target = input shifted by 1)
        end = end_idx if end_idx < len(tokenized) else (end_idx - 1)
        input_chunk = tokenized[start_idx:end]
        target_chunk = tokenized[start_idx+1:end+1]
        
        # Pad if necessary (input and target always have same length, so pad the same amount)
        pad_length = self.max_length - len(input_chunk)
        if pad_length > 0:
            # Use numpy padding for efficiency (int32 to match tokenized dtype)
            padding = np.full(pad_length, self.pad_token, dtype=tokenized.dtype)
            input_chunk = np.concatenate([input_chunk, padding])
            target_chunk = np.concatenate([target_chunk, padding])
        
        # Convert directly from numpy to tensor (no intermediate list conversion)
        # Input can stay int32 (saves memory, embeddings accept int32)
        # Target MUST be int64 (cross_entropy requires long/int64 for labels)
        input_tensor = torch.from_numpy(input_chunk)  # int32 - saves memory
        target_tensor = torch.from_numpy(target_chunk.astype(np.int64))  # int64 - required for cross_entropy
        return input_tensor, target_tensor
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        assert idx >= 0, "idx must be non-negative"
        assert idx < len(self.sequences), "idx must be less than the number of sequences"
        
        # Get sequence from index
        file_idx, start_idx, end_idx = self.sequences[idx]
        tokenized = self._get_tokenized_file(file_idx)
        input_seq, target_seq = self._get_sequence(tokenized, start_idx, end_idx)
        
        return input_seq, target_seq

class StreamingDataLoader:
    """Custom DataLoader wrapper for StreamingDataset."""
    
    def __init__(self, tokenizer, 
                 max_length: int, stride: int, pad_token: int, # Dataset related parameters
                 batch_size: int, shuffle: bool = True, drop_last: bool = False, # Dataloader related parameters
                 num_workers: int = 0, mode: Literal['train', 'val'] = 'train', 
                 train_ratio: float = 0.95, eval_num_batches: int = None,
                 pin_memory: bool = True, # Controls whether batches are allocated in pinned (page-locked) memory for faster CPU→GPU transfers
                 prefetch_factor: int = 2, # Controls how many batches each worker prefetches ahead of time
                 force_build: bool = False): # Force rebuild cache even if it exists
        self.batch_size = batch_size
        self.eval_num_batches = eval_num_batches
        
        # Dataset
        self.dataset = StreamingDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            pad_token=pad_token,
            train_ratio=train_ratio,
            mode=mode
        )
        
        # Dataloader
        # persistent_workers=True when num_workers > 0 (faster for multi-epoch training)
        # Workers stay alive between epochs, keeping caches warm
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )

        # Eval dataloader if sampling enabled
        self.sample_dataloader = None
        if self.eval_num_batches is not None:
            self.sample_dataloader = self.create_eval_loader(self.eval_num_batches)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def next_batch(self):
        return next(iter(self))
    
    def create_eval_loader(self, num_batches: int) -> DataLoader:
        # Calculate number of samples needed for num_batches
        total_batches = len(self.dataloader)
        num_samples = min(num_batches, total_batches) * self.batch_size
        
        # Create RandomSampler to sample random samples (without replacement)
        sampler = RandomSampler(self.dataset, num_samples=num_samples, replacement=False)
        
        # Create DataLoader with random sampler, reusing original settings
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=self.dataloader.drop_last,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
            prefetch_factor=getattr(self.dataloader, 'prefetch_factor', None),
            persistent_workers=True if self.dataloader.num_workers > 0 else False
        )

def train_model(model, tokenizer, optimizer: torch.optim.Optimizer,
                train_loader: StreamingDataLoader,
                val_loader: StreamingDataLoader,
                device: torch.device,
                num_epochs: int, eval_freq: int,
                warmup_steps: int, initial_lr: float = 3e-5, min_lr: float = 1e-6,
                grad_clip_norm: float = 1.0,
                train_eval_batches: int = 10, val_eval_batches: int = 10,
                save_path: str = get_data_config().model_dir + "/pretraining/", # Directory to save checkpoints
                checkpoint_freq: int = 50000, # Frequency to save checkpoints
                verbose: bool = False) -> tuple[list[float], list[float], list[int]]:

    # Initialize training metrics and step variables
    # Per step metrics
    losses, retrieval_times, compute_times, lr_values = [], [], [], []
    tokens_seen, global_step = 0, -1
    # Per eval round metrics
    train_losses, val_losses, track_tokens_seen, eval_times, eval_steps = [], [], [], [], []
    
    # Print token statistics
    print_token_statistics(train_loader, val_loader)
    
    # Start time
    start_time = time.perf_counter()
    
    # Reset GPU memory peak counter at start of training and verbose GPU info
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    
    # Setup checkpoint, steps, and plot directories
    dir_path = get_data_config().model_dir + "/pretraining/" + save_path
    checkpoint_dir = dir_path + "/checkpoints" if checkpoint_freq > 0 else None
    plot_dir = dir_path + "/plots"
    steps_dir = dir_path + "/steps"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(steps_dir, exist_ok=True)   
    # Warmup steps and cosine decay (learning rate scheduler) parameters
    total_steps = num_epochs * len(train_loader) # Total number of training steps
    max_lr = optimizer.param_groups[0]['lr'] # Maximum learning rate
    lr_increment = (max_lr - initial_lr)/warmup_steps # Learning rate increment per step
    
    # Verbose logging
    if verbose:
        print(f"Training model for {num_epochs} epochs and {total_steps} total steps")
    
    # Single progress bar tracking total steps
    pbar = tqdm(total=total_steps, desc="Training", unit="step")
    data_start = time.perf_counter()
    
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Iterate over dataloader batches
        for input_batch, target_batch in train_loader.dataloader:
            
            # Ensure batch from dataloader is moved to correct device
            if input_batch.device != device:
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
            
            # Update batch retrieval times
            retrieval_time = time.perf_counter() - data_start
            retrieval_times.append(retrieval_time)
                
            # Record step (forward + backward pass) start time
            compute_start_time = time.perf_counter()
            
            # Set model to training mode
            model.train()
            # Reset loss gradients
            optimizer.zero_grad()

            # Learning rate
            # Warmup
            global_step += 1
            if global_step <= warmup_steps:
                lr = initial_lr + (global_step * lr_increment)
            # Cosine decay
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress)))
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            lr_values.append(lr)

            # Forward pass
            loss = calculate_batch_loss(
                model=model,
                input_batch=input_batch, 
                target_batch=target_batch, 
                device=device,
                pad_token=pad_token
            )
            
            # Backward pass
            loss.backward() # Compute gradients
            
            # Gradient clipping
            if grad_clip_norm > 0.0:
                clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            # Update parameters
            optimizer.step()
            
            # Update step (forward + backward pass) metrics
            tokens_seen += input_batch.numel()
            step_compute_time = time.perf_counter() - compute_start_time
            compute_times.append(step_compute_time)
            
            # Calculate metrics for progress bar
            current_loss = loss.item()
            losses.append(current_loss)
            
            pbar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{current_loss:.3f}',
                'tokens': format_tokens(tokens_seen)
            })
            pbar.update(1)
            
            # Verbose logging
            if verbose:
                print(f"{'='*60}")
                print(f"Epoch {epoch+1} - Step {global_step:06d}:")
                print(f"    Global step: {global_step:06d} of {total_steps:06d}")
                print(f"    Batch retrieval time: {retrieval_time:.3f}s")
                print(f"    Step (forward + backward pass) time: {step_compute_time:.3f}s")
                print(f"    Total time: {retrieval_time + step_compute_time:.3f}s")
                print(f"{'='*60}")
            
            # Optional evaluation step (don't eval on step 0)
            if global_step % eval_freq == 0:
                eval_start_time = time.perf_counter()
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    pad_token=pad_token,
                    train_eval_batches=train_eval_batches,
                    val_eval_batches=val_eval_batches,
                    verbose=verbose
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                eval_steps.append(global_step)
                eval_duration = time.perf_counter() - eval_start_time
                eval_times.append(eval_duration)

                # Timing metrics
                avg_step_time = sum(compute_times) / len(compute_times)
                avg_eval_time = sum(eval_times) / len(eval_times)
                last_step_time = compute_times[-1]
                total_elapsed_time = time.perf_counter() - start_time
                est_time_remaining = avg_step_time * (total_steps - global_step)
                est_time_remaining += avg_eval_time * ((total_steps - global_step) / eval_freq)
                eta = start_time + est_time_remaining
                eta_h, eta_m, eta_s = convert_time(eta)
                
                print(
                    f"\n{'='*60}\n"
                    f"Training metrics: Epoch {epoch+1} - Step {global_step:06d}:\n"
                    f"{'='*60}\n"
                    f"    Global step:       {global_step:06d} of {total_steps:06d}\n"
                    f"    Train loss:        {train_loss:.3f} ({train_eval_batches} batches)\n"
                    f"    Val loss:          {val_loss:.3f} ({val_eval_batches} batches)\n"
                    f"    Tokens seen:       {format_tokens(tokens_seen)} ({tokens_seen:,})\n"
                    f"    Current step:      {last_step_time:.3f}s\n"
                    f"    Avg step time:     {avg_step_time:.3f}s\n"
                    f"    Avg eval time:     {avg_eval_time:.3f}s\n"
                    f"    Total elapsed:     {total_elapsed_time:.1f}s\n"
                    f"    ETA to completion: {eta_h}h {eta_m}m {eta_s}s\n"
                    f"{'='*60}\n"
                )

                # Print GPU stats if available (verbose only)
                if device.type == 'cuda':
                    get_gpu_stats(device)

                # Generate sample text
                generate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_tokens=50
                )

                # Log per-step metrics
                log_steps(
                    steps_dir=steps_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    tokens_seen=track_tokens_seen[-eval_freq:],
                    losses=losses[-eval_freq:],
                    retrieval_times=retrieval_times[-eval_freq:],
                    compute_times=compute_times[-eval_freq:],
                    lr_values=lr_values[-eval_freq:],
                )
                # Log per-eval metrics
                log_eval(
                    steps_dir=steps_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr_last=lr,
                    eval_time=eval_duration,
                    total_elapsed_s=total_elapsed_time,
                    eta_hms=f"{eta_h}h{eta_m}m{eta_s}s",
                )
            
            # Optional checkpoint saving
            if checkpoint_freq > 0 and global_step > 0 and global_step % checkpoint_freq == 0:
                # Model checkpoint
                save_checkpoint(
                    model=model, 
                    checkpoint_dir=checkpoint_dir, 
                    global_step=global_step, 
                    verbose=True
                )

                # Plot loss up to current step
                plot_loss(
                    global_step=global_step,
                    step_losses=losses,
                    eval_steps=eval_steps,
                    eval_tokens=track_tokens_seen,
                    eval_train_losses=train_losses,
                    eval_val_losses=val_losses,
                    save_path=plot_dir,
                )

            # Start timing for next batch retrieval (for troubleshooting)
            # This measures the gap between steps which includes batch retrieval time
            data_start = time.perf_counter()
    
    pbar.close()  # Close progress bar after training    

    # Save final model checkpoint
    save_checkpoint(model, checkpoint_dir, verbose=True)
    
    # Evaluate model on final step
    eval_start_time = time.perf_counter()
    final_train_loss, final_val_loss = evaluate_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            pad_token=pad_token,
            train_eval_batches=train_eval_batches,
            val_eval_batches=val_eval_batches
    )
    train_losses.append(final_train_loss)
    val_losses.append(final_val_loss)
    track_tokens_seen.append(tokens_seen)
    eval_steps.append(global_step)
    eval_duration = time.perf_counter() - eval_start_time
    eval_times.append(eval_duration)

    # Final loss plot
    plot_loss(
        global_step=global_step,
        step_losses=losses,
        eval_steps=eval_steps,
        eval_tokens=track_tokens_seen,
        eval_train_losses=train_losses,
        eval_val_losses=val_losses,
        save_path=plot_dir,
    )

    # Final step metrics
    steps_since_last_eval = global_step - eval_steps[-1] if len(eval_steps) > 0 else global_step + 1
    steps_to_slice = min(steps_since_last_eval, len(losses))
    log_steps(
        steps_dir=steps_dir,
        epoch=num_epochs,
        global_step=global_step,
        tokens_seen=track_tokens_seen[-steps_to_slice:],
        losses=losses[-steps_to_slice:],
        retrieval_times=retrieval_times[-steps_to_slice:],
        compute_times=compute_times[-steps_to_slice:],
        lr_values=lr_values[-steps_to_slice:],
    )
    # Final eval metrics
    log_eval(
        steps_dir=steps_dir,
        epoch=epoch + 1,
        global_step=global_step,
        tokens_seen=tokens_seen,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        lr_last=lr,
        eval_time=eval_duration,
        total_elapsed_s=total_elapsed_time,
        eta_hms=f"{eta_h}h{eta_m}m{eta_s}s",
    )
    
    # Print final training summary
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total tokens processed: {format_tokens(tokens_seen)} ({tokens_seen:,})")
    print(f"Total steps: {global_step}")
    print(f"Total epochs: {num_epochs}")
    print(f"Final train loss: {final_train_loss:.3f}")
    print(f"Final val loss: {final_val_loss:.3f}")
    print(f"{'='*60}\n")

    return losses, train_losses, val_losses, track_tokens_seen

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GPT2 Model Pretraining Configuration")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
    parser.add_argument("--eval_freq", type=int, default=500,
                        help="Frequency of evaluations during training")
    parser.add_argument("--max_lr", type=float, default=5e-4,
                        help="Maximum learning rate (used as max_lr in cosine decay schedule)")
    parser.add_argument("--initial_lr", type=float, default=3e-5,
                        help="Initial learning rate for warmup")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay (L2 regularization) for optimizer")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of DataLoader workers (None=auto: 0 for Windows, 2-4 for Linux)")
    parser.add_argument("--checkpoint_freq", type=int, default=25000,
                        help="Frequency of checkpoint saves (in steps). Set to 0 to disable checkpoints.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output (detailed timing, batch info, GPU stats, generated samples)")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Uses a very small model for debugging purposes")

    args = parser.parse_args()

    # Configure DataLoader workers
    num_workers = configure_dataloader_workers(user_specified=args.num_workers, verbose=True)
    
    # Device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)

    # Model config
    if args.debug:
        GPT_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_len": 10,        # Context length
            "embed_dim": 12,          # Embedding dimension
            "num_heads": 2,          # Number of attention heads
            "num_layers": 2,         # Number of layers
            "drop_emb_rate": 0.0,    # Embedding dropout rate
            "drop_attn_rate": 0.0,   # Attention dropout rate
            "drop_ffn_rate": 0.0,    # Feedforward dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_len": 1024,     # Context length
            "embed_dim": 768,        # Embedding dimension
            "num_heads": 12,         # Number of attention heads
            "num_layers": 12,        # Number of layers
            "drop_emb_rate": 0.1,   # Embedding dropout rate
            "drop_attn_rate": 0.1,   # Attention dropout rate
            "drop_ffn_rate": 0.1,   # Feedforward dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    # Model and tokenizer
    model = GPTModel(GPT_CONFIG).to(device)
    print(f"Model config: {GPT_CONFIG}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    tokenizer = tiktoken.get_encoding("gpt2")
    # Determine pad_token from tokenizer
    pad_token = 50256  # GPT2 EOT token - TODO: extract from tokenizer

    # Optimizer (lr will be overridden by train_model's learning rate schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    
    # Create custom data loaders
    # Files are discovered automatically by StreamingDataset from config
    print("Creating training data loader...")
    train_loader = StreamingDataLoader(
        tokenizer=tokenizer,
        max_length=GPT_CONFIG["context_len"],
        stride=GPT_CONFIG["context_len"],
        pad_token=pad_token,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        mode='train',  # Training mode
        eval_num_batches=30,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    print("Creating validation data loader...")
    val_loader = StreamingDataLoader(
        tokenizer=tokenizer,
        max_length=GPT_CONFIG["context_len"],
        stride=GPT_CONFIG["context_len"],
        pad_token=pad_token,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        mode='val',  # Validation mode
        eval_num_batches=30,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Train model (verbose logging already in the train_model function)
    losses, train_losses, val_losses, tokens_seen = train_model(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        warmup_steps=1000,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        grad_clip_norm=1.0,
        train_eval_batches=150,
        val_eval_batches=150,
        save_path="gpt2-small",
        checkpoint_freq=args.checkpoint_freq,
        verbose=args.verbose
    )

    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
