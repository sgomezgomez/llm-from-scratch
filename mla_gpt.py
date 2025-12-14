import torch, torch.nn as nn, torch.nn.functional as F
# Switched to KV cached multi head attention
# from attention import MultiHeadAttention_ScaledDotProduct

class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["embed_dim"])
        self.pos_emb = nn.Embedding(self.cfg["context_len"], self.cfg["embed_dim"])
        self.drop_emb = nn.Dropout(self.cfg["drop_emb_rate"])
        # Switched to KV cached multi head attention
        #self.transformer_blocks = nn.Sequential(*[TransformerBlock(self.cfg) for _ in range(self.cfg["num_layers"])])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg["num_layers"])])
        self.final_norm = LayerNorm(self.cfg["embed_dim"])
        self.out_head = nn.Linear(self.cfg["embed_dim"], self.cfg["vocab_size"], bias=False)

        # KV cache current position pointer
        self.ptr_current_position = 0
    
    def forward(self, x: torch.Tensor, kv_cache: bool = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        token_emb = self.token_emb(x)
        # Switched to KV cached multi head attention
        # pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device)) # device setting allos us to train model on GPT or CPU
        if kv_cache:
            pos_ids = torch.arange(self.ptr_current_position, (self.ptr_current_position + seq_len), device=x.device, dtype=torch.long)
            self.ptr_current_position += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0)

        x = token_emb + pos_emb
        x = self.drop_emb(x)
        # Switched to KV cached multi head attention
        # x = self.transformer_blocks(x)
        for block in self.transformer_blocks:
            x = block(x, kv_cache=kv_cache) # KV cached multi head attention
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

        for _ in range(max_tokens):
            # Generate logits for next token
            cropped_idx = idx[:, -self.cfg["context_len"]:] # (batch, context_len)
            # With KV caching, we only project the next token id after the first prediction (not the entire sequence)
            if _ > 0:
                cropped_idx = idx[:, -1:]
            with torch.no_grad():
                logits = self(cropped_idx, kv_cache=True)
            logits = logits[:, -1, :] # (batch, n_tokens, vocab_size) --> (batch, vocab_size)
            # Negative infinity
            neg_inf = torch.finfo(logits.dtype).min
            
            # Greedy sampling
            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            else:
            # Top-k sampling
                if top_k is not None and 0 < top_k < logits.shape[-1]:
                    top_logits, _ = torch.topk(logits, top_k)
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
        self.attention = MultiHeadLatentAttention(
            d_in=cfg["embed_dim"],
            d_out=cfg["embed_dim"],
            context_len=cfg["context_len"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_attn_rate"],
            bias=cfg["qkv_bias"],
            # Latent dimension
            latent_dim=cfg["latent_dim"]
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embed_dim"])
        self.norm2 = LayerNorm(cfg["embed_dim"])
        self.drop_attn_shortcut = nn.Dropout(cfg["drop_attn_rate"])
        self.drop_ffn_shortcut = nn.Dropout(cfg["drop_ffn_rate"])
    
    def forward(self, x, kv_cache: bool = False):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x, kv_cache=kv_cache) # KV cached multi head attention
        x = self.drop_attn_shortcut(x)
        x = x + shortcut # Residual connection
        
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_ffn_shortcut(x)
        x = x + shortcut # Residual connection
        return x

# Variation of the multi head attention using KV caching and latent attention (MLA)
# Adapted from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/05_mla/gpt_with_kv_mla.py
# Cited source includes: https://huggingface.co/bird-of-paradise/deepseek-mla
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 context_len: int, 
                 dropout: float, 
                 num_heads: int, 
                 bias: bool = False, 
                 latent_dim: int = None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce projection dim to match desired output dim
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)
        # Combine queries, keys, and values into a single linear layer
        self.w_q = nn.Linear(d_in, d_out, bias=bias)
        self.dkv = nn.Linear(d_in, self.latent_dim, bias=bias) # Projection down to latent_dim
        # Fused key-value projection: single linear layer outputs both keys and values
        self.w_ukv = nn.Linear(self.latent_dim, 2 * d_out, bias=bias) # Latent -> keys and values
        self.proj = nn.Linear(d_out, d_out) # Linear projection to combine head outputs
        self.dropout = dropout
        # KV cache buffers
        self.register_buffer("cache_c_kv", None, persistent=False)
    
    def forward(self, x: torch.Tensor, kv_cache: bool = False) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape

        # Project input into queries and new latent chunk
        queries = self.w_q(x) # (batch, num_tokens, d_out)
        latent_new = self.dkv(x) # (batch, num_tokens, latent_dim)

        # Update latent cache and choose latent sequence to up-project
        if kv_cache:
            if self.cache_c_kv is None or self.cache_c_kv.size(0) != batch:
                self.cache_c_kv = torch.zeros(batch, self.context_len, self.latent_dim, device=x.device) # (batch, context_len, latent_dim)
            # If incoming tokens exceed context length, shift cache left
            if self.ptr_current_position + num_tokens > self.context_len:
                overflow = self.ptr_current_position + num_tokens - self.context_len
                self.cache_c_kv[:, :-overflow, :] = self.cache_c_kv[:, overflow:, :].clone()
                self.ptr_current_position -= overflow
            # Update cache with new tokens
            self.cache_c_kv[:, self.ptr_current_position:self.ptr_current_position+num_tokens, :] = latent_new
            self.ptr_current_position += num_tokens
            # Use cached latent
            latent_total = self.cache_c_kv[:, :self.ptr_current_position, :]
        else:
            latent_total = latent_new
        
        #p-project latent to keys and values (fused operation)
        kv = self.w_ukv(latent_total) # (batch, T_k_total, 2 * d_out)
        keys, values = kv.chunk(2, dim=-1) # Split into keys and values (batch, T_k_total, d_out)

        # Reshape to heads
        q = self._reshape_to_heads(queries, self.num_heads, self.head_dim)
        k = self._reshape_to_heads(keys, self.num_heads, self.head_dim)
        v = self._reshape_to_heads(values, self.num_heads, self.head_dim)

        Tq = q.shape[-2] # Number of queries in current batch, Also happens to be the number of tokens in current batch
        T_k_total = k.shape[-2] # Number of keys/values in total (cache + current tokens)

        # If no KV cache or number of keys/values and queries match
        # (During training, no KV cache)
        # (Also during inference, even with KV cache, when the number of keys/values and queries match -- first pass)
        if kv_cache is None or not kv_cache or (T_k_total == Tq):
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=(self.dropout if self.training else 0.0)
            )
        # Only one token in current batch -- No causal mask or dropout
        # (During inference, with kv cache after first pass)
        # q: (batch, num_heads, 1, head_dim)
        # k, v: (batch, num_heads, T_k_total, head_dim)
        elif Tq == 1:
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=0.0
            )
        # (During batched inference with KV cache)
        else:
            # Attention mask: True = keep, False = mask
            prefix_len = T_k_total - Tq # Number of tokens to mask
            # shift the 1s diagonal to the right by prefix_len
            attn_mask = torch.zeros(Tq, T_k_total, dtype=torch.bool, device=q.device) # True = keep, False = mask
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            # Create causal mask only for the new tokens (more efficient than full mask)
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            context_vec = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # Mask handles causality, avoid double masking
                dropout_p=0.0
            )

        # Combine heads
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.proj(context_vec)
        return context_vec
    
    def reset_kv_cache(self):
        self.cache_c_kv = None
        self.ptr_current_position = 0
    
    @staticmethod
    def _reshape_to_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        batch, seq_len, d_out = x.shape
        return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim)) # Gamma
        self.shift = nn.Parameter(torch.zeros(embed_dim)) # Beta
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.04475 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    cfg = {
        "vocab_size": 50257,
        "context_len": 1024,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "qkv_bias": False,
        "drop_emb_rate": 0.1,
        "drop_ffn_rate": 0.1,
        "drop_attn_rate": 0.1,
        "latent_dim": 512,
    }
    model = GPTModel(cfg)
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

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, how are you?"
    input_ids = torch.tensor([tokenizer.encode(text, allowed_special="all")], dtype=torch.long) # Convert list to tensor and add batch dimension
    output = model.generate_simple(input_ids, 10)
    print(tokenizer.decode(output.squeeze(0).tolist()))
