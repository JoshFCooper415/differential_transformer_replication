import torch
import torch.nn as nn
from torch.nn import functional as F

class GroupLayerNorm(nn.Module):
    """Group Layer Normalization for independent head normalization"""
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.eps = 1e-5
        self.num_heads = num_heads
        self.head_dim = head_dim * 2  # Account for double head size
        self.weight = nn.Parameter(torch.ones(1, 1, num_heads * self.head_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, num_heads * self.head_dim))

    def forward(self, x):
        # x shape: (batch, seq_len, num_heads * head_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias

class DiffHead(nn.Module):
    """One head of differential attention"""
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key1 = nn.Linear(n_embd, head_size, bias=False)
        self.query1 = nn.Linear(n_embd, head_size, bias=False)
        self.key2 = nn.Linear(n_embd, head_size, bias=False)
        self.query2 = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size * 2, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
        # Learnable lambda parameters
        self.lambda_q1 = nn.Parameter(torch.zeros(head_size))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_size))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_size))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_size))
        self.register_buffer('lambda_init', torch.tensor(0.8))

    def get_lambda(self, layer_idx):
        layer_idx_tensor = torch.tensor(layer_idx, dtype=torch.float, device=self.lambda_init.device)
        dynamic_init = 0.8 - 0.6 * torch.exp(-0.3 * (layer_idx_tensor - 1.0))
        self.lambda_init.copy_(dynamic_init)
        lambda_val = torch.exp(self.lambda_q1 * self.lambda_k1) - \
                    torch.exp(self.lambda_q2 * self.lambda_k2) + \
                    self.lambda_init
        return lambda_val.mean()

    def forward(self, x, layer_idx):
        B, T, C = x.shape
        
        k1, q1 = self.key1(x), self.query1(x)
        k2, q2 = self.key2(x), self.query2(x)
        v = self.value(x)
        
        scale = 1.0 / (k1.shape[-1] ** 0.5)
        att1 = (q1 @ k1.transpose(-2, -1)) * scale
        att2 = (q2 @ k2.transpose(-2, -1)) * scale
        
        att1 = att1.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att2 = att2.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)
        att1 = self.dropout(att1)
        att2 = self.dropout(att2)
        
        lambda_val = self.get_lambda(layer_idx)
        diff_att = att1 - lambda_val * att2
        
        out = diff_att @ v
        return out

class MultiHeadDiffAttention(nn.Module):
    """Multiple heads of differential attention in parallel"""
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            DiffHead(head_size, n_embd, dropout, block_size) 
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
    """Swish-Gated Linear Unit"""
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // (n_head * 2)  # Halved due to differential attention
        self.diff_attn = MultiHeadDiffAttention(
            n_head, head_size, n_embd, dropout, block_size
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

class DiffTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx