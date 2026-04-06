import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_HAND_FEATURES = 27  # lh_1...lh_27 and rh_1...rh_27


# ==========================================
# BUILDING BLOCKS
# ==========================================

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
        return F.relu(x + self.block(x))


class FeatureAttention(nn.Module):
    """Self-attention over features to learn which features matter most."""
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
        x_seq = x.unsqueeze(1)                        # (batch, 1, dim)
        attended, _ = self.attention(x_seq, x_seq, x_seq)
        attended = attended.squeeze(1)                # (batch, dim)
        return self.norm(x + attended)


# ==========================================
# HAND PATHWAYS
# ==========================================

class DominantHandPath(nn.Module):
    """
    LARGER pathway for the dominant (more affected) hand.
    3 layers, wider hidden dim.
    
    Why larger: PD tremor is stronger in dominant/affected hand,
    so we give the model more capacity to extract patterns from it.
    """
    def __init__(self, in_features, out_dim, dropout=0.3):
        super().__init__()
        self.path = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.path(x)  # (batch, out_dim)


class NonDominantHandPath(nn.Module):
    """
    SMALLER pathway for the non-dominant hand.
    2 layers, narrower hidden dim.
    
    Why smaller: Less tremor signal here for PD,
    less capacity needed.
    """
    def __init__(self, in_features, out_dim, dropout=0.3):
        super().__init__()
        self.path = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.path(x)  # (batch, out_dim)


class AsymmetryPath(nn.Module):
    """
    Small pathway for asymmetry features (abs difference between hands).
    
    Why: Asymmetry between hands is a key PD diagnostic marker.
    Healthy: left ≈ right → asymmetry ≈ 0
    PD:      one hand worse → asymmetry >> 0
    """
    def __init__(self, in_features, out_dim, dropout=0.3):
        super().__init__()
        self.path = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.path(x)  # (batch, out_dim)


# ==========================================
# MAIN MODEL
# ==========================================

class TremorClassifier(nn.Module):
    """
    Dual-pathway tremor classifier for Healthy vs PD.

    Architecture:
        1. Split input into left (lh) and right (rh) hand features
        2. Determine dominant hand per sample using handedness
        3. Route dominant hand → LARGER pathway
        4. Route non-dominant hand → SMALLER pathway
        5. Compute asymmetry (abs diff) → ASYMMETRY pathway
        6. Concatenate all 3 pathways + embeddings
        7. Self-attention → residual blocks → classification

    Input (from CSV):
        features:   (batch, 312) = [lh_1...lh_156, rh_1...rh_156]
        handedness: (batch,)     - 0=Left-handed, 1=Right-handed
        movement:   (batch,)     - movement ID 0-10

    Output:
        logits:     (batch, 1)   - binary classification
    """
    def __init__(
        self,
        num_hand_features=NUM_HAND_FEATURES,  # 156 per hand
        num_movements=11,
        num_classes=1,

        # Embedding dimensions
        movement_embed_dim=32,
        handedness_embed_dim=8,

        # Each pathway outputs this dim before fusion
        pathway_out_dim=128,

        # Fusion network
        hidden_dim=256,
        num_residual_blocks=3,
        num_attention_heads=8,

        dropout=0.3
    ):
        super().__init__()

        self.num_hand_features = num_hand_features

        # ---- Embeddings ----
        self.movement_embed = nn.Embedding(num_movements, movement_embed_dim)
        self.handedness_embed = nn.Embedding(2, handedness_embed_dim)

        # ---- Hand Pathways ----
        self.dominant_path = DominantHandPath(num_hand_features, pathway_out_dim, dropout)
        self.nondominant_path = NonDominantHandPath(num_hand_features, pathway_out_dim, dropout)
        self.asymmetry_path = AsymmetryPath(num_hand_features, pathway_out_dim // 2, dropout)

        # ---- Fusion projection ----
        # dominant(128) + nondominant(128) + asymmetry(64) + movement(32) + handedness(8)
        fusion_input_dim = (
            pathway_out_dim +          # dominant path
            pathway_out_dim +          # non-dominant path
            pathway_out_dim // 2 +     # asymmetry path
            movement_embed_dim +
            handedness_embed_dim
        )

        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---- Attention ----
        self.attention = FeatureAttention(hidden_dim, num_heads=num_attention_heads)

        # ---- Residual Blocks ----
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _split_features(self, features):
        """
        Split (batch, 312) into left and right hand features.
        
        Returns:
            lh: (batch, 156) - left hand features
            rh: (batch, 156) - right hand features
        """
        lh = features[:, :self.num_hand_features]   # lh_1...lh_156
        rh = features[:, self.num_hand_features:]   # rh_1...rh_156
        return lh, rh

    def _route_hands(self, lh, rh, handedness):
        """
        Route left/right hand to dominant/non-dominant pathway
        based on handedness.
        
        handedness: 0 = Left-handed (left hand is dominant)
                    1 = Right-handed (right hand is dominant)
        
        Returns:
            dominant:     (batch, 156)
            nondominant:  (batch, 156)
        """
        # handedness mask: True where right-handed (dominant = right)
        is_right_handed = handedness.bool().unsqueeze(1)  # (batch, 1)

        # Right-handed → right hand is dominant
        # Left-handed  → left hand is dominant
        dominant = torch.where(is_right_handed, rh, lh)
        nondominant = torch.where(is_right_handed, lh, rh)

        return dominant, nondominant

    def forward(self, features, handedness, movement):
        """
        Args:
            features:   (batch, 312) - [lh_1...lh_156, rh_1...rh_156]
            handedness: (batch,)     - 0=left-handed, 1=right-handed
            movement:   (batch,)     - movement ID

        Returns:
            logits: (batch, 1)
        """
        # 1. Split into left and right hand features
        lh, rh = self._split_features(features)

        # 2. Compute asymmetry BEFORE routing
        asymmetry = torch.abs(lh - rh)  # (batch, 156)

        # 3. Route to dominant / non-dominant based on handedness
        dominant, nondominant = self._route_hands(lh, rh, handedness)

        # 4. Process each pathway separately
        dom_out   = self.dominant_path(dominant)     # (batch, 128)
        nondom_out = self.nondominant_path(nondominant)  # (batch, 128)
        asym_out  = self.asymmetry_path(asymmetry)   # (batch, 64)

        # 5. Embed categorical variables
        movement_emb  = self.movement_embed(movement)        # (batch, 32)
        handedness_emb = self.handedness_embed(handedness)   # (batch, 8)

        # 6. Fuse all pathways + embeddings
        x = torch.cat([
            dom_out,
            nondom_out,
            asym_out,
            movement_emb,
            handedness_emb
        ], dim=1)  # (batch, 128+128+64+32+8 = 360)

        # 7. Project to hidden dim
        x = self.fusion_projection(x)  # (batch, 256)

        # 8. Self-attention
        x = self.attention(x)  # (batch, 256)

        # 9. Residual blocks
        for block in self.residual_blocks:
            x = block(x)  # (batch, 256)

        # 10. Classify
        logits = self.classifier(x)  # (batch, 1)

        return logits