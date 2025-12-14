import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
from huggingface_hub import login, hf_hub_download
from config.config import access_config
from typing import List

class Llama2Model(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["embed_dim"], dtype=self.cfg["dtype"])
        # No need for positional or dropout embedding for Llama 2
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(self.cfg) for _ in range(self.cfg["num_layers"])])
        self.final_norm = RMSNorm(self.cfg["embed_dim"], dtype=self.cfg["dtype"])
        self.out_head = nn.Linear(self.cfg["embed_dim"], self.cfg["vocab_size"], dtype=self.cfg["dtype"], bias=False)
        # KV cache current position pointer
        self.ptr_current_position = 0
        # Precompute Rotary Positional Embedding (RoPE) for faster inference
        # At model level to avoid recomputing for each attention head
        cos, sin = self.precompute_rope()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def precompute_rope(self, theta_base=10_000):
        # Compute inverse frequencies
        # Shape: (head_dim//2,)
        head_dim = self.cfg["embed_dim"] // self.cfg["num_heads"]
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=self.cfg["dtype"], device=self.get_device())[:(head_dim//2)] / head_dim))
        # Generate position indices
        # Shape: (context_len,)
        pos = torch.arange(self.cfg["context_len"], dtype=self.cfg["dtype"], device=self.get_device())
        # Compute angles
        # (context_len, 1) * (1, head_dim//2) -> (context_len, head_dim//2)
        angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0) 
        # Expand angles to match head_dim
        angles = torch.cat([angles, angles], dim=1) # (context_len, head_dim)
        # Precompute cosine and sine
        cos, sin = torch.cos(angles), torch.sin(angles) # (context_len, head_dim) -- each

        return cos, sin
    
    def forward(self, x: torch.Tensor, kv_cache: bool = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        token_emb = self.token_emb(x)
        # No positional embedding for llama 2

        # Switched to KV cached multi head attention
        if kv_cache:
            self.ptr_current_position += seq_len

        x = token_emb
        # Switched to KV cached multi head attention
        # x = self.transformer_blocks(x)
        for block in self.transformer_blocks:
            # KV cached multi head attention with RoPE
            # Adjust cos and sin shapes to match the sequence length (seq_len)
            x = block(x, self.cos[:seq_len, :], self.sin[:seq_len, :], kv_cache=kv_cache) 
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.attention.reset_kv_cache()
        self.ptr_current_position = 0
    
    # Simple generation function, now supporting KV caching for inference
    def generate_simple(self, idx: torch.Tensor, max_tokens: int, kv_cache: bool = True) -> torch.Tensor:
        self.reset_kv_cache()
        self.eval()
        for _ in range(max_tokens):
            cropped_idx = idx[:, -self.cfg["context_len"]:]
            # With KV caching, we only project the next token id after the first prediction (not the entire sequence)
            if kv_cache and _ > 0:
                cropped_idx = idx[:, -1:]
            with torch.no_grad():
                logits = self(cropped_idx, kv_cache=kv_cache)
            
            logits = logits[:, -1, :] # (batch, n_tokens, vocab_size) --> (batch, vocab_size)
            # Logits --> Probabilities (with softmax)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            # Append new token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def generate(self, idx: torch.Tensor,
                 max_tokens: int,
                 temperature: float = 1.0,
                 top_k: int = None, 
                 top_p: float = None,
                 end_of_text_token: int = None) -> torch.Tensor:
        # Assertions
        assert isinstance(idx, torch.Tensor), "idx must be a tensor"
        assert idx.dim() == 2, "idx must be a 2D tensor"
        self.eval()
        self.reset_kv_cache()

        for step_idx in range(max_tokens):
            # Generate logits for next token
            cropped_idx = idx[:, -self.cfg["context_len"]:] # (batch, context_len)
            # With KV caching, we only project the next token id after the first prediction (not the entire sequence)
            if step_idx > 0:
                cropped_idx = idx[:, -1:]
            with torch.no_grad():
                logits = self(cropped_idx, kv_cache=True)  # Always use KV cache during generation
            logits = logits[:, -1, :] # (batch, n_tokens, vocab_size) --> (batch, vocab_size)
            # Negative infinity
            neg_inf = torch.finfo(logits.dtype).min
            
            # Greedy sampling
            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            else:
                # Top-k sampling
                if top_k is not None and 0 < top_k < logits.shape[-1]:
                    top_logits, top_indices_k = torch.topk(logits, top_k)
                    min_val = top_logits[:, [-1]] # Kth treshold per batch
                    logits = logits.masked_fill(logits < min_val, neg_inf)
                
                logits = logits / temperature # Temperature needs to be greater than 0
                # Logits --> Probabilities (with softmax)
                probs = F.softmax(logits, dim=-1)
                
                # Top-p (nucleus) sampling
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Find cutoff index where cumulative exceeds top_p (more efficient than torch.where)
                    # Expand top_p to match batch dimension for searchsorted
                    top_p_tensor = torch.full((cum_probs.shape[0], 1), top_p, device=cum_probs.device, dtype=cum_probs.dtype)
                    cutoff = torch.searchsorted(cum_probs, top_p_tensor, right=False).clamp(min=1, max=sorted_probs.shape[-1])
                    # Create mask: keep positions before cutoff
                    # If cumulative never exceeds top_p, cutoff will be vocab_size and we keep all tokens
                    keep = torch.arange(sorted_probs.shape[-1], device=sorted_probs.device).unsqueeze(0).expand_as(sorted_probs) < cutoff
                    sorted_probs = sorted_probs.masked_fill(~keep, 0.0)
                    # Renormalize
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    # Map back to original positions using scatter
                    probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
                
                # Sample from probabilities
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append new token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            # Yield next token instead of waiting to return the entire sequence
            yield idx_next

            # Stop if end of text token is reached
            if end_of_text_token is not None and torch.any(idx_next == end_of_text_token):
                break

    def get_device(self):
        return next(self.parameters()).device
    
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.get_device()))
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention_KVCache(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_len=cfg["context_len"],
            num_heads=cfg["num_heads"],
            dtype=cfg["dtype"],
            # KV cache parameters
            max_seq_len=cfg["max_seq_len"] if "max_seq_len" in cfg else None,
            window_size=cfg["window_size"] if "window_size" in cfg else None
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["embed_dim"], dtype=cfg["dtype"])
        self.norm2 = RMSNorm(cfg["embed_dim"], dtype=cfg["dtype"])
        # No shortcut (residual connection) dropout for Llama 2
    
    def forward(self, x, cos, sin, kv_cache: bool = False):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, cos, sin, kv_cache=kv_cache) # KV cached multi head attention
        # No shortcut (residual connection) dropout for Llama 2
        x = x + shortcut # Residual connection
        
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        # No shortcut (residual connection) dropout for Llama 2
        x = x + shortcut # Residual connection
        return x

# Variation of the multi head attention using KV caching
# Useful to speed up inference
# Adapted from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/gpt_with_kv_cache_optimized.py
# Also adapted from: https://github.com/karpathy/nanochat/blob/a83646e098374de4d4f2c2da1175d2f5fdd18ed3/nanochat/gpt.py
class MultiHeadAttention_KVCache(nn.Module):
    def __init__(self, 
                 d_in: int, # Embedding dimension
                 d_out: int, # Hidden dimension
                 context_len: int,
                 # No dropout for llama 2 
                 num_heads: int, 
                 dtype: torch.dtype, 
                 max_seq_len: int = None, 
                 window_size: int = None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in # Embedding dimension
        self.d_out = d_out # Hidden dimension
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce projection dim to match desired output dim
        self.context_len = context_len
        self.dtype = dtype # Store dtype for cache initialization
        # Combine queries, keys, and values into a single linear layer
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=False, dtype=dtype)
        self.proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype) # Linear projection to combine head outputs
        # No dropout for llama 2
        # KV cache buffers
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.max_seq_len = max_seq_len or context_len
        self.window_size = window_size or self.max_seq_len # Sliding window size for KV caching
    
    def apply_rope(self, x, cos, sin):
        # x corresponds to the projected queries and keys
        batch, num_heads, num_tokens, head_dim = x.shape
        assert cos.shape == (num_tokens, self.head_dim), "cos and sin shape must match the sequence length"
        assert sin.shape == cos.shape, "cos and sin must have the same shape"
        # Split x into halves
        x1 = x[..., :head_dim//2] # First half -> (batch, num_heads, num_tokens, head_dim//2
        x2 = x[..., head_dim//2:] # Second half -> (batch, num_heads, num_tokens, head_dim//2)
        # Adjust cos and sin shapes
        cos = cos.unsqueeze(0).unsqueeze(0) # (num_tokens, head_dim) -> (1, 1, num_tokens, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0) # (num_tokens, head_dim) -> (1, 1, num_tokens, head_dim)
        # Apply rotary transformation
        # This simulates multiplying by [0 -1; 1 0] rotation matrix
        rotated = torch.cat((-x2, x1), dim=-1) # (batch, num_heads, num_tokens, head_dim)
        # Apply formula
        x_rotated = (x * cos) + (rotated * sin) # (batch, num_heads, num_tokens, head_dim)

        return x_rotated.to(dtype=x.dtype)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, kv_cache: bool = False) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape
        # Project input into queries, keys, and values --> (batch, num_tokens, 3 * d_in)
        qkv = self.qkv(x)
        # With KV caching, we only project the next token id after the first prediction (not the entire sequence)
        # This means num_tokens == Tq == 1 in those cases
        qkv = qkv.view(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, batch, num_heads, num_tokens, head_dim)
        q, k_new, v_new = qkv # 3 times (batch, num_heads, num_tokens, head_dim)
        # Apply RoPE -- QK rotary embedding
        q = self.apply_rope(q, cos, sin) # (batch, num_heads, num_tokens, head_dim)
        k_new = self.apply_rope(k_new, cos, sin) # (batch, num_heads, num_tokens, head_dim)

        if kv_cache:
            # Initialize cache as zeros tensors
            if self.cache_k is None or self.cache_k.size(0) != batch:
                self.cache_k = torch.zeros(
                    batch, self.num_heads, self.window_size, self.head_dim, 
                    device=x.device, 
                    dtype=self.dtype
                )
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_current_position = 0 # # Pointer to the next free position in the cache
            # If incoming tokens exceed window size, shift cache left
            if self.ptr_current_position + num_tokens > self.window_size: # Overflow occurs when new tokens exceed window size
                overflow = self.ptr_current_position + num_tokens - self.window_size # Number of tokens to shift left
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                self.ptr_current_position -= overflow # Pointer after shift
            # Update cache with new tokens
            self.cache_k[:, :, self.ptr_current_position:self.ptr_current_position+num_tokens, :] = k_new
            self.cache_v[:, :, self.ptr_current_position:self.ptr_current_position+num_tokens, :] = v_new
            self.ptr_current_position += num_tokens
            # Use cached keys and values
            k = self.cache_k[:, :, :self.ptr_current_position, :]
            v = self.cache_v[:, :, :self.ptr_current_position, :]
            
        else:
            k, v = k_new, v_new
        Tq = q.shape[-2] # Number of queries in current batch, Also happens to be the number of tokens in current batch
        Tk = k.shape[-2] # Number of keys/values in total (cache + current tokens)

        # If no KV cache or number of keys/values and queries match
        # (During training, no KV cache)
        # (Also during inference, even with KV cache, when the number of keys/values and queries match -- first pass)
        if kv_cache is None or not kv_cache or (Tk == Tq):
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True
                # No dropout for llama 2
            )
        # Only one token in current batch -- No causal mask or dropout
        # (During inference, with kv cache after first pass)
        # q: (batch, num_heads, 1, head_dim)
        # k, v: (batch, num_heads, window_size, head_dim)
        elif Tq == 1:
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False
            )
        # (During batched inference with KV cache)
        else:
            # Attention mask: True = keep, False = mask
            prefix_len = Tk - Tq # Number of tokens to mask
            # shift the 1s diagonal to the right by prefix_len
            attn_mask = torch.zeros(Tq, Tk, dtype=torch.bool, device=q.device) # True = keep, False = mask
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask
            )

        # Combine heads
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec
    
    def reset_kv_cache(self):
        self.cache_k, self.cache_v = None, None

# RMSNorm instead of LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5, dtype=None):
        super().__init__()
        # Option 1: pytorch native RMSNorm
        self.rmsnorm = torch.nn.RMSNorm(normalized_shape=embed_dim, eps=eps, dtype=dtype)

        # Option 2: custom RMSNorm
        # self.eps = eps
        # self.embed_dim = embed_dim
        # self.weight = nn.Parameter(torch.ones(embed_dim, dtype=dtype))
    
    def forward(self, x):
        # Option 1: pytorch native RMSNorm
        return self.rmsnorm(x)
        # Option 2: custom RMSNorm
        # Cast to float32 for stability
        # x_fp32 = x.float() 
        # means = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt effectively replaces 1/sqrt
        # x_normed = x_fp32 * torch.rsqrt(means + self.eps)
        # Cast back to original dtype
        # return (x_normed * self.weight).to(dtype=x.dtype)
  
# SiLU instead of GELU activation
# Represents the custom SiLU implementation
# Note: This is not used in the Llama2 model, but is included for completeness
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

# Llama uses a Gates Linear nit (GLU) variat of silu called SwiGLU
# This results in a different structure for the feedforward layer
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["embed_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["embed_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["embed_dim"], dtype=cfg["dtype"], bias=False)
        # We created a custom SiLU class for demo purposes, but we'll use the native PyTorch SiLU
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2 # PyTorch native SiLU
        return self.fc3(x)

class Llama2Tokenizer:
    def __init__(self, model_name: str = "Llama-2-7b-hf"):
        access_token = access_config.huggingface_api_key
        login(access_token)
        tokenizer_file = hf_hub_download(
            repo_id="meta-llama/" + model_name,
            filename="tokenizer.model",
            local_dir=model_name
        )
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_file)
        self.tokenizer = sp
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    cfg = {
        "vocab_size": 32000,
        "context_len": 4096,
        "embed_dim": 4096,
        "hidden_dim": 11008,
        "num_heads": 32,
        "num_layers": 32,
        "dtype": torch.float16,
    }
    tokenizer = Llama2Tokenizer()
    model = Llama2Model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params * 4 / 1024 / 1024
    total_params_feedforward = sum(p.numel() for block in model.transformer_blocks for p in block.ffn.parameters())
    total_params_attention = sum(p.numel() for block in model.transformer_blocks for p in block.attention.parameters())
    print(f"Total model parameters: {total_params}")
    print(f"Feedforward parameters: {total_params_feedforward}")
    print(f"Attention parameters: {total_params_attention}")
    print(f"Model size: {total_params_mb:.2f} MB")
    print(f"Token embedding shape: {model.token_emb.weight.shape}")
    print(f"Ouput layer shape: {model.out_head.weight.shape}")
    
    # Test with different sequence lengths to show flexibility
    test_lengths = [10, 50, 100, 500, 1024]
    
    for seq_len in test_lengths:
        x = torch.randint(0, cfg["vocab_size"], (1, seq_len))
        logits = model(x)
        print(f"Sequence length {seq_len:4d}: Input shape {x.shape} -> Output shape {logits.shape}")

    text = "Hello, how are you?"
    input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long) # Convert list to tensor and add batch dimension
    output = model.generate_simple(input_ids, 10)
    print(tokenizer.decode(output.squeeze(0).tolist()))
