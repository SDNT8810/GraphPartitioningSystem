"""Attention-based Q-Network for Proposed_Method system."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionQNetwork(nn.Module):
    """Attention-based Q-Network for graph partitioning."""
    
    def __init__(self, node_features, hidden_dim=64, num_heads=4, dropout=0.1):
        """Initialize the attention Q-network.
        
        Args:
            node_features (int): Number of input node features
            hidden_dim (int): Hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # Q-value output
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, adj_matrix):
        """Forward pass of the network.
        
        Args:
            node_features (torch.Tensor): Node features [batch_size, num_nodes, node_features]
            adj_matrix (torch.Tensor): Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            torch.Tensor: Q-values for each node [batch_size, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Project input
        x = self.input_proj(node_features)  # [batch_size, num_nodes, hidden_dim]
        
        # Multi-head attention
        attention_mask = (adj_matrix.sum(dim=-1) != 0).float()
        x2, _ = self.attention(x, x, x, key_padding_mask=~attention_mask.bool())
        x = x + self.dropout(x2)
        x = self.layer_norm1(x)
        
        # Feed forward
        x2 = self.fc2(F.relu(self.fc1(x)))
        x = x + self.dropout(x2)
        x = self.layer_norm2(x)
        
        # Output Q-values
        q_values = self.out(x).squeeze(-1)  # [batch_size, num_nodes]
        
        # Mask invalid nodes
        q_values = q_values.masked_fill(~attention_mask.bool(), float('-inf'))
        
        return q_values
