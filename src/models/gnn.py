import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config.system_config import GNNConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Split into heads
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn_scores = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim ** 0.5)
        
        # Apply edge mask
        row, col = edge_index
        attn_mask = torch.zeros((x.size(0), x.size(0)), device=x.device)
        attn_mask[row, col] = 1
        attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1) == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        out = torch.einsum('nhm,mhd->nhd', attn_weights, v)
        out = out.reshape(-1, self.hidden_channels)
        out = self.out_proj(out)
        
        return out

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        scale = (2 ** (self.bits - 1) - 1) / x.abs().max()
        return torch.round(x * scale) / scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.quantize(self.weight)
        bias = self.quantize(self.bias)
        return F.linear(x, weight, bias)

class GNNLayer(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int, dropout: float = 0.1, 
                 use_attention: bool = True, quantize: bool = False, bits: int = 8):
        super().__init__()
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = MultiHeadAttention(hidden_channels, num_heads, dropout)
        
        if quantize:
            self.lin = QuantizedLinear(hidden_channels, hidden_channels, bits)
        else:
            self.lin = nn.Linear(hidden_channels, hidden_channels)
            
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            x = x + self.attention(x, edge_index)
        else:
            x = x + self.lin(x)
            
        x = self.norm(x)
        x = self.dropout(x)
        return x

class GraphNeuralNetwork(nn.Module):
    def __init__(self, config: GNNConfig, input_dim=None):
        super().__init__()
        self.config = config
        
        # Input dimension will be determined dynamically on first forward pass if not provided
        self.input_dim = input_dim
        self.input_proj = None  # Will be initialized in the first forward pass
        
        self.layers = nn.ModuleList([
            GNNLayer(
                hidden_channels=config.hidden_channels,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_attention=config.use_attention,
                quantize=config.quantize,
                bits=config.quantization_bits
            ) for _ in range(config.num_layers)
        ])
        
        self.final_proj = nn.Linear(config.hidden_channels, config.num_partitions)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Initialize input projection dynamically based on actual input dimension
        if self.input_proj is None:
            input_dim = x.shape[1]
            self.input_proj = nn.Linear(input_dim, self.config.hidden_channels).to(x.device)
            print(f"Initializing input projection: {input_dim} -> {self.config.hidden_channels}")
        
        # Project input features to hidden dimension
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.final_proj(x)