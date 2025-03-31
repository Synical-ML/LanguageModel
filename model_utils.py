import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


@dataclass
class GPTConfig:
    max_length: int
    num_layers: int
    num_heads: int
    d_model: int
    rate: float
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        self.vocab_size = len(self.tokenizer.get_vocab())


class PolyReLU(nn.Module):
    def __init__(self, order=3, init_coeff=0.0):
        super(PolyReLU, self).__init__()
        if order < 2:
            raise ValueError("Order must be at least 2.")
        self.order = order

        self.coeffs = nn.Parameter(torch.full((order - 1,), init_coeff, dtype=torch.float))

    def forward(self, x):
        out = x
        for i in range(self.order - 1):
            out = out + self.coeffs[i] * (x ** (i + 2))
        return out


class PolyNorm(nn.Module):
    def __init__(self, normalized_shape, order=3, eps=1e-5, init_coeff=0.0):
        super(PolyNorm, self).__init__()
        if order < 2:
            raise ValueError("Order must be at least 2.")
        self.order = order
        self.eps = eps

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

        self.coeffs = nn.Parameter(torch.full((order - 1,), init_coeff, dtype=torch.float))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)

        poly_out = normed
        for i in range(self.order - 1):
            poly_out = poly_out + self.coeffs[i] * (normed ** (i + 2))

        return self.gamma * poly_out + self.beta


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp_in_proj = nn.Linear(config.d_model, config.d_model * 4)
        self.mlp_out_proj = nn.Linear(config.d_model * 4, config.d_model)
        self.mlp_activation = PolyReLU()
        self.dropout = nn.Dropout(config.rate)

    def forward(self, x):
        return self.dropout(self.mlp_out_proj(self.mlp_activation(self.mlp_in_proj(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv_w = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.rate)

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads

        mask = torch.tril(torch.ones(config.max_length, config.max_length))
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_w(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self.mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn = torch.nn.functional.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return self.dropout(out)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = PolyNorm(config.d_model)
        self.ln2 = PolyNorm(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding_layer = nn.Embedding(config.vocab_size, config.d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.lm_head(x)

        return x
