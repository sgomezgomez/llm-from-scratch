import torch, torch.nn as nn, torch.nn.functional as F

class SelfAttention_vParam(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.randn(d_in, d_out))
        self.W_k = nn.Parameter(torch.randn(d_in, d_out))
        self.W_v = nn.Parameter(torch.randn(d_in, d_out))
    
    def forward(self, x):
        queries = x @ self.W_q
        keys = x @ self.W_k
        values = x @ self.W_v
        att_scores = queries @ keys.transpose(-2, -1)  # .transpose(-2, -1) instead of .T
        att_weights = F.softmax(att_scores/(keys.shape[-1]**0.5), dim=-1)
        context_vec = att_weights @ values
        return context_vec

class SelfAttention_vLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
    
    def forward(self, x):
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        att_scores = queries @ keys.transpose(-2, -1)  # .transpose(-2, -1) instead of .T
        att_weights = F.softmax(att_scores/(keys.shape[-1]**0.5), dim=-1)
        context_vec = att_weights @ values
        return context_vec

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # The use of register_buffer in PyTorch is not strictly necessary but offers 
        # advantages such as moving buffers to the appropriate device (CPU vs. GPU)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))
    
    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        att_scores = queries @ keys.transpose(-2, -1)
        att_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        att_weights = F.softmax(att_scores/(keys.shape[-1]**0.5), dim=-1)
        att_weights = self.dropout(att_weights)
        context_vec = att_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttention(d_in, d_out, context_len, dropout, bias) 
        for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce projection dim to match desired output dim
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.output_proj = nn.Linear(d_out, d_out) # Linear projection to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))
    
    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        # Input shape: (batch, num_tokens, d_in)
        # Project input into queries, keys, and values
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        
        # Split queries, keys, and values into heads and transpose
        # We implicitly split the matrix by adding a num_heads dimension
        # (batch, num_tokens, d_out) --> (batch, num_tokens, num_heads, head_dim)
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        att_scores = queries @ keys.transpose(-2, -1) # Dot product for each head --> (batch, num_heads, num_tokens, num_tokens)
        att_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        att_weights = F.softmax(att_scores/(keys.shape[-1]**0.5), dim=-1)
        att_weights = self.dropout(att_weights)

        # Apply attention weights to values to compute context vectors
        context_vec = (att_weights @ values).transpose(1, 2) # (batch, num_heads, num_tokens, head_dim)
        # Combine heads
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out) # Where self.d_out = self.num_heads * self.head_dim
        context_vec = self.output_proj(context_vec) # Optional linear projection
        return context_vec # Output shape: (batch, num_tokens, d_out)

class MultiHeadAttention_ScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.context_len = context_len
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce projection dim to match desired output dim
        # Combine queries, keys, and values into a single linear layer
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.proj = nn.Linear(d_out, d_out) # Linear projection to combine head outputs
        self.dropout = dropout
    
    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        
        # Input shape: (batch, num_tokens, d_in)
        # Project input into queries, keys, and values --> (batch, num_tokens, 3 * d_in)
        qkv = self.qkv(x)
        qkv = qkv.view(batch, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, batch, num_heads, num_tokens, head_dim)

        q, k, v = qkv # 3 times (batch, num_heads, num_tokens, head_dim)

        # Pytorch's scaled dot product attention
        context_vec = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=True, 
            dropout_p=self.dropout
        ).transpose(1, 2)
        
        # Combine heads
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out) # (batch, num_tokens, d_out)

        context_vec = self.proj(context_vec) # Optional linear projection
        return context_vec # Output shape: (batch, num_tokens, d_out)

if __name__ == "__main__":
    # Let's use d_in ≠ d_out for comparison
    d_in = 3
    d_out = 4
    batch_size = 2
    context_len = 6
    dropout = 0.
    num_heads = 2
    
    # Self Attention
    x = torch.randn(batch_size, context_len, d_in)
    self_attention = SelfAttention_vParam(d_in, d_out)
    self_attention_linear = SelfAttention_vLinear(d_in, d_out)
    print(f"Compare:\n{self_attention(x) == self_attention_linear(x)}")
    
    # Using weights from linear into the param class
    print("\nUsing weights from linear into the param class...")
    print(f"W_q shape from param:\n{self_attention.W_q.shape}")  # should be [8, 12]
    print(f"W_q shape from linear:\n{self_attention_linear.W_q.weight.shape}")  # should be [12, 8]
    self_attention.W_q.data = self_attention_linear.W_q.weight.transpose(-2, -1)
    self_attention.W_k.data = self_attention_linear.W_k.weight.transpose(-2, -1)
    self_attention.W_v.data = self_attention_linear.W_v.weight.transpose(-2, -1)
    print(f"Compare:\n{self_attention(x) == self_attention_linear(x)}")

    # Causal Self Attention
    causal_self_attention = CausalSelfAttention(d_in, d_out, context_len=context_len, dropout=dropout)
    print(f"\nCausal Self Attention:\n{causal_self_attention(x)}")

    # Multi Head Attention
    multi_head_attention_wrapper = MultiHeadAttentionWrapper(d_in, d_out, context_len=context_len, dropout=dropout, num_heads=num_heads)
    print(f"\nMulti Head Attention Wrapper:\n{multi_head_attention_wrapper(x)}")
    multi_head_attention = MultiHeadAttention(d_in, d_out, context_len=context_len, dropout=dropout, num_heads=num_heads)
    print(f"\nMulti Head Attention:\n{multi_head_attention(x)}")
    multi_head_attention_scaled_dot_product = MultiHeadAttention_ScaledDotProduct(d_in, d_out, context_len=context_len, dropout=dropout, num_heads=num_heads)
    print(f"\nMulti Head Attention Scaled Dot Product:\n{multi_head_attention_scaled_dot_product(x)}")
    print(f"Compare:\n{multi_head_attention(x) == multi_head_attention_scaled_dot_product(x)}")