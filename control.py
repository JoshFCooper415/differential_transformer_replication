import torch
import torch.nn as nn
import torch.nn.functional as F
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex rotation"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensors"""
    # Ensure these are the same device
    freqs_cis = freqs_cis.to(x.device)
    
    # Reshape for broadcasting
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1], :]
    
    # Apply rotation using complex multiplication
    x_rotated = torch.view_as_real(x_complex * freqs_cis.unsqueeze(0)).flatten(-2)
    return x_rotated.type_as(x)

class Head(nn.Module):
    """One head of self-attention with RoPE"""
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
        # Register RoPE frequency buffer
        freqs_cis = precompute_freqs_cis(head_size, block_size)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        B, T, C = x.shape
        
        # Compute key, query, value projections
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Apply rotary embeddings
        k = apply_rotary_emb(k, self.freqs_cis)
        q = apply_rotary_emb(q, self.freqs_cis)

        # Compute attention scores
        scale = 1.0 / (k.shape[-1] ** 0.5)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, T, T)
        
        # Mask future positions (for decoder)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        out = att @ v  # (B, T, head_size)
        return out
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, dropout, block_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (unchanged)"""
    def __init__(self, size_in, size_out):
        super().__init__()
        self.linear_gate = nn.Linear(size_in, size_out)
        self.linear_xform = nn.Linear(size_in, size_out)
        
    def forward(self, x):
        gate = F.silu(self.linear_gate(x))
        xform = self.linear_xform(x)
        return gate * xform

class Block(nn.Module):
    """Transformer block (unchanged except for initialization)"""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(
            n_head, head_size, n_embd, dropout, block_size
        )
        self.ffwd = nn.Sequential(
            SwiGLU(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        # Remove position embedding table, keep only token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Only use token embeddings, no position embeddings needed
        x = self.token_embedding_table(idx)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx