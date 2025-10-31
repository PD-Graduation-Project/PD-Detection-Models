import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class TremorResNet(nn.Module):
    def __init__(self, num_classes=3, num_movements=11):
        super().__init__()
        
        self.conv1 = nn.Conv1d(6, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, 2, 1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Metadata
        self.wrist_embed = nn.Embedding(2, 32)
        self.movement_embed = nn.Embedding(num_movements, 64)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x, wrist, movement):
        # x: [B, T, 6] -> [B, 6, T]
        x = x.transpose(1, 2)
        
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)
        
        # Add metadata
        wrist_emb = self.wrist_embed(wrist.long())
        movement_emb = self.movement_embed(movement.long())
        
        combined = torch.cat([x, wrist_emb, movement_emb], dim=1)
        
        return self.classifier(combined)