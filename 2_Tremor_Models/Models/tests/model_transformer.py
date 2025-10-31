import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    

class TremorTransformer(nn.Module):
    def __init__(self, num_classes=3, num_movements=11, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(6, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Wrist and movement embeddings
        self.wrist_embed = nn.Embedding(2, d_model//4)
        self.movement_embed = nn.Embedding(num_movements, d_model//4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model//2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, wrist, movement):
        # x: [B, T, 6]
        B, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # [B, T, d_model]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]
        
        # Add wrist and movement info
        wrist_emb = self.wrist_embed(wrist.long())  # [B, d_model//4]
        movement_emb = self.movement_embed(movement.long())  # [B, d_model//4]
        
        combined = torch.cat([x, wrist_emb, movement_emb], dim=1)
        
        return self.classifier(combined)

