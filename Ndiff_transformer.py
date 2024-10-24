import torch
import torch.nn as nn
from torch.nn import functional as F
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

class GroupLayerNorm(nn.Module):
    """Group Layer Normalization for independent head normalization"""
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.eps = 1e-5
        self.num_heads = num_heads
        self.head_dim = head_dim * 2  # Double head size for information capacity
        self.weight = nn.Parameter(torch.ones(1, 1, num_heads * self.head_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_heads * self.head_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class AlternatingDiffHead(nn.Module):
    """Differential attention with RoPE and natural N-term extension"""
    def __init__(self, head_size, n_embd, dropout, block_size, n_terms=2):
        super().__init__()
        self.n_terms = n_terms
        self.head_size = head_size
        self.block_size = block_size
        
        # Create query and key projections
        self.queries = nn.ModuleList([
            nn.Linear(n_embd, head_size, bias=False)
            for _ in range(n_terms)
        ])
        self.keys = nn.ModuleList([
            nn.Linear(n_embd, head_size, bias=False)
            for _ in range(n_terms)
        ])
        
        # Double value projection (unchanged)
        self.value = nn.Linear(n_embd, head_size * 2, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
        # Per-dimension lambda parameters (unchanged)
        self.lambda_qs = nn.ParameterList([
            nn.Parameter(torch.zeros(head_size))
            for _ in range(n_terms)
        ])
        self.lambda_ks = nn.ParameterList([
            nn.Parameter(torch.zeros(head_size))
            for _ in range(n_terms)
        ])
        
        # Register RoPE frequency buffer
        freqs_cis = precompute_freqs_cis(head_size, block_size)
        self.register_buffer("freqs_cis", freqs_cis)
        
        self.register_buffer('lambda_init', torch.tensor(0.8))

    def get_lambda(self, layer_idx):
        """Calculate lambda values for all terms (unchanged)"""
        layer_idx_tensor = torch.tensor(layer_idx, dtype=torch.float, device=self.lambda_init.device)
        dynamic_init = 0.8 - 0.6 * torch.exp(-0.3 * (layer_idx_tensor - 1.0))
        self.lambda_init.copy_(dynamic_init)
        
        lambda_vals = []
        for i in range(self.n_terms):
            lambda_val = torch.exp(self.lambda_qs[i] * self.lambda_ks[i])
            if i > 0:
                lambda_val = lambda_val - torch.exp(self.lambda_qs[i-1] * self.lambda_ks[i-1])
            lambda_val = lambda_val + self.lambda_init
            lambda_vals.append(lambda_val.mean())
            
        return torch.stack(lambda_vals)

    def forward(self, x, layer_idx):
        B, T, C = x.shape
        v = self.value(x)
        scale = 1.0 / (self.head_size ** 0.5)
        
        # Calculate all attention maps with RoPE
        attention_maps = []
        for i in range(self.n_terms):
            # Apply rotary embeddings to queries and keys
            q = self.queries[i](x)
            k = self.keys[i](x)
            
            # Apply RoPE to queries and keys
            q_rotated = apply_rotary_emb(q, self.freqs_cis)
            k_rotated = apply_rotary_emb(k, self.freqs_cis)
            
            att = (q_rotated @ k_rotated.transpose(-2, -1)) * scale
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            attention_maps.append(att)
        
        # Combine attention maps (unchanged)
        lambda_vals = self.get_lambda(layer_idx)
        diff_att = attention_maps[0] * lambda_vals[0]
        
        for i in range(1, self.n_terms):
            sign = -1 if i % 2 else 1
            diff_att = diff_att + sign * lambda_vals[i] * attention_maps[i]
        
        out = diff_att @ v
        return out

class MultiHeadAlternatingDiffAttention(nn.Module):
    """Multiple heads of alternating differential attention"""
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size, n_terms=2):
        super().__init__()
        self.heads = nn.ModuleList([
            AlternatingDiffHead(head_size, n_embd, dropout, block_size, n_terms)
            for _ in range(num_heads)
        ])
        self.group_norm = GroupLayerNorm(num_heads, head_size)
        self.proj = nn.Linear(head_size * 2 * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('lambda_init', torch.tensor(0.8))

    def forward(self, x, layer_idx):
        out = torch.cat([h(x, layer_idx) for h in self.heads], dim=-1)
        out = self.group_norm(out)
        out = out * (1 - self.lambda_init)
        out = self.dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation"""
    def __init__(self, size_in, size_out):
        super().__init__()
        self.linear_gate = nn.Linear(size_in, size_out)
        self.linear_xform = nn.Linear(size_in, size_out)
        
    def forward(self, x):
        gate = F.silu(self.linear_gate(x))
        xform = self.linear_xform(x)
        return gate * xform

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head, block_size, dropout, n_terms=4):
        super().__init__()
        head_size = n_embd // (n_head * 2)  # Halved due to double head size
        self.diff_attn = MultiHeadAlternatingDiffAttention(
            n_head, head_size, n_embd, dropout, block_size, n_terms
        )
        self.ffwd = nn.Sequential(
            SwiGLU(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, layer_idx):
        x = x + self.diff_attn(self.ln1(x), layer_idx)
        x = x + self.ffwd(self.ln2(x))
        return x

class AlternatingDiffTransformer(nn.Module):
    """Complete Alternating Differential Transformer with RoPE"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, n_terms=4):
        super().__init__()
        self.block_size = block_size
        
        # Remove position embedding table, only keep token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout, n_terms)
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
        
        # Only use token embeddings, no position embeddings
        x = self.token_embedding_table(idx)
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks, 1):
            x = block(x, i)
        
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
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @staticmethod
    def from_pretrained(path):
        """Load model from checkpoint"""
        config = torch.load(path)
        model = AlternatingDiffTransformer(**config['model_args'])
        model.load_state_dict(config['model_state'])
        return model

    def save_pretrained(self, path):
        """Save model checkpoint"""
        config = {
            'model_args': {
                'vocab_size': self.token_embedding_table.num_embeddings,
                'n_embd': self.token_embedding_table.embedding_dim,
                'n_head': len(self.blocks[0].diff_attn.heads),
                'n_layer': len(self.blocks),
                'block_size': self.block_size,
                'dropout': self.blocks[0].diff_attn.dropout.p,
                'n_terms': self.blocks[0].diff_attn.heads[0].n_terms,
            },
            'model_state': self.state_dict()
        }
        torch.save(config, path)