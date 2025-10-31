import torch
from torch import nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.bottleneck = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
            for kernel_size, padding in [(10, 4), (20, 9), (40, 19)]
        ])
        
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        
    def forward(self, x):
        # Bottleneck
        x = self.bottleneck(x)
        
        # Parallel convolutions
        branches = [conv(x) for conv in self.convs]
        branches.append(self.maxpool(x))
        
        # Concatenate and batch norm
        x = torch.cat(branches, dim=1)
        return F.relu(self.bn(x))

class TremorInceptionTime(nn.Module):
    def __init__(self, num_classes=3, num_movements=11, in_channels=6, hidden_channels=32):
        super().__init__()
        
        # Inception modules
        self.inception_blocks = nn.Sequential(
            InceptionModule(in_channels, hidden_channels),
            InceptionModule(hidden_channels * 4, hidden_channels * 2),
            InceptionModule(hidden_channels * 8, hidden_channels * 4),
        )
        
        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Wrist and movement embeddings
        self.wrist_embed = nn.Embedding(2, 32)
        self.movement_embed = nn.Embedding(num_movements, 64)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 16 + 32 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, wrist, movement):
        # x: [B, T, 6] -> [B, 6, T]
        x = x.transpose(1, 2)
        
        # Inception blocks
        x = self.inception_blocks(x)
        
        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Add wrist and movement
        wrist_emb = self.wrist_embed(wrist.long())
        movement_emb = self.movement_embed(movement.long())
        
        combined = torch.cat([x, wrist_emb, movement_emb], dim=1)
        
        return self.classifier(combined)