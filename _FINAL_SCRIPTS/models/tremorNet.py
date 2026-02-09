import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with batch norm and dropout.
    
    Input → Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → (+Input) → ReLU
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        # Residual connection
        return F.relu(x + self.block(x))


class FeatureAttention(nn.Module):
    """
    Self-attention over features.
    Learns which features are important for classification.
    """
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, feature_dim)
        Returns:
            attended: (batch, feature_dim)
        """
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # Self-attention
        attended, _ = self.attention(x, x, x)  # (batch, 1, feature_dim)
        
        # Remove sequence dimension
        attended = attended.squeeze(1)  # (batch, feature_dim)
        
        # Residual + normalization
        return self.norm(x.squeeze(1) + attended)


class TremorClassifier(nn.Module):
    """
    Advanced feature-based classifier with:
        - Categorical embeddings (movement, handedness)
        - Feature attention
        - Residual blocks
        - Batch normalization
    
    Architecture:
        1. Embed categorical variables
        2. Project features + embeddings to common dimension
        3. Apply self-attention over features
        4. Stack residual blocks
        5. Classification head
    """
    def __init__(
        self,
        num_features=66,           # Left + Right + Asymmetry features
        num_movements=11,          # Number of movement types
        num_classes=1,             # Healthy vs PD -> with one preceptron only
        
        # Embedding dimensions
        movement_embed_dim=32,
        handedness_embed_dim=8,
        
        # Network dimensions
        hidden_dim=256,            # Main processing dimension
        num_residual_blocks=3,     # How many residual blocks
        num_attention_heads=8,     # Attention heads
        
        # Regularization
        dropout=0.3
    ):
        super().__init__()
        
        # ==========================================
        # 1. EMBEDDINGS for categorical variables
        # ==========================================
        self.movement_embed = nn.Embedding(num_movements, movement_embed_dim)
        self.handedness_embed = nn.Embedding(2, handedness_embed_dim)
        
        # ==========================================
        # 2. PROJECTION to common dimension
        # ==========================================
        total_input = num_features + movement_embed_dim + handedness_embed_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==========================================
        # 3. ATTENTION over features
        # ==========================================
        self.attention = FeatureAttention(
            feature_dim=hidden_dim,
            num_heads=num_attention_heads
        )
        
        # ==========================================
        # 4. RESIDUAL BLOCKS
        # ==========================================
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # ==========================================
        # 5. CLASSIFICATION HEAD
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features, handedness, movement):
        """
        Args:
            features: (batch, num_features) - tremor features
            handedness: (batch,) - 0 or 1
            movement: (batch,) - movement ID
        
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = features.size(0)
        
        # 1. Embed categorical variables
        movement_emb = self.movement_embed(movement)        # (batch, 32)
        handedness_emb = self.handedness_embed(handedness)  # (batch, 8)
        
        # 2. Concatenate all inputs
        x = torch.cat([features, movement_emb, handedness_emb], dim=1)
        
        # 3. Project to hidden dimension
        x = self.input_projection(x)  # (batch, hidden_dim)
        
        # 4. Apply attention
        x = self.attention(x)  # (batch, hidden_dim)
        
        # 5. Pass through residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)  # (batch, hidden_dim)
        
        # 6. Classification
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits