import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        weight_tensor = torch.empty((out_features, in_features), device = device, dtype = dtype)
        self.W = nn.Parameter(weight_tensor)
        std = math.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
    def forward(self, x):
        return x@self.W.T

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device = None, dtype=None):
        super().__init__()
        weight_tensor = torch.empty((num_embedding, embedding_dim), device=device, dtype=dtype)
        self.embedding = nn.Parameter(weight_tensor)
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)
    def forward(self, idx):
        return self.embedding[idx]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    def forward(self, x):
        in_dtype=x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        return (self.weight*inv_rms*x).to(in_dtype)

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        pos = torch.arange(max_seq_len, dtype=torch.float32, device = device)
        j = torch.arange(0, d_k//2, dtype=torch.float32, device=device)
        inv_freq = theta**(-2.0*j/d_k)
        freqs = torch.outer(pos, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None)->torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        elif token_positions.ndim == 2:
            token_positions = token_positions.unsqueeze(1)
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot  = x_even * sin + x_odd * cos
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        return x_rot.reshape(x.shape)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.device = device
        self.d_k = d_model//num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    def forward(self, x, token_positions=None):
        batch, seq, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        Q = Q.reshape(batch, self.num_heads, seq, self.d_k)
        K = K.reshape(batch, self.num_heads, seq, self.d_k)
        scores = (Q@K.transpose(-2, -1) / math.sqrt(d_model//self.num_heads))
        mask = torch.tril(torch.ones((seq, seq), device = x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights@V
        concat_out = attn_out.transpose(1,2).contiguous().view(batch, seq, d_model)
        return self.output_proj(concat_out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = Linear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = Linear(d_model, d_ff, device = device, dtype = dtype)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device = None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device = device, dtype = dtype)
        self.ln1 = RMSNorm(d_model, eps=1e-5, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device = device, dtype = dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5, device = device, dtype = dtype)
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)
        h = x + attn_out
        h_norm = self.ln2(h)
        ffn_out = self.ffn(h_norm)
        return h+ffn_out

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device = None, dtype=None)for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(d_model, device = None, dtype=None)
        self.lm_head = Linear(d_model, vocab_size, device = None, dtype=None)
    def forward(self, indices):
        x = self.token_embeddings(indices)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)