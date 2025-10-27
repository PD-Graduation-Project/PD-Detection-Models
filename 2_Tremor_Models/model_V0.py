import torch
from torch import nn

class TremorNetGRU(nn.Module):
    def __init__(self,
                num_classes: int = 3,
                hidden_size: int = 128,
                wrist_embed_dim:int = 16,
                dropout:float = 0.3,
                ):
        super().__init__()
        
        # 1. CNN feature extractor
        # --------------------------
        # Input: [B, C=6, T] -> after 3 conv stride-2 layers -> approx T/8
        self.cnn = nn.Sequential(
            # 1.1. Conv layer 1: 6 -> 64 channels
            nn.Conv1d(
                in_channels= 6, # 6 IMU channels
                out_channels= 64,
                kernel_size= 5, stride=2, padding=2), # T -> T/2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 1.2. Conv layer 2: 64 -> 128 channels
            nn.Conv1d(
                in_channels= 64,
                out_channels= 128,
                kernel_size= 3, stride=2, padding=1), # T/2 -> T/4
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 1.3. Conv layer 3: 128 -> 256 channels (captures finer motion patterns)
            nn.Conv1d(
                in_channels= 128,
                out_channels= 256,
                kernel_size=3, stride=2, padding=1),  # T/4 -> T/8
            nn.BatchNorm1d(256),
            nn.ReLU(),

            
            nn.Dropout(dropout)
        )
        
        # 2. GRU for temporal modeling (time-series data)
        # -------------------------------------------------
        self.gru = nn.GRU(
            input_size= 256, # input from CNN: 256 features per time step (after downsampling)
            hidden_size= hidden_size,
            num_layers=2, # stacking 2 GRU together
            batch_first= True,
            bidirectional= True, # outputs features in both time directions -> doubles hidden size
            dropout= dropout # applied between GRU layers (not after the last one)
        )
        
        # 3. Attention mechanism (focuses on important time windows)
        # -----------------------------------------------------------
        # applied on the GRU outputs [B, T', hidden_size*2]
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Embedding wrist (left/ right)
        # ---------------------------------
        self.wrist_embed = nn.Embedding(
            num_embeddings= 2, # 0 = Left, 1 = Right
            embedding_dim= wrist_embed_dim 
            # each wrist type is represented by a learned 'wrist_embed_dim'-dimensional vector instead of a single scalar.
        ) # Output shape -> [B, wrist_embed_dim]
        
        # 5. Classifier
        # ---------------
        # input dim = (attended + mean + max) = 3 * (hidden_size*2) = hidden_size*6
        # plus wrist_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*6 + wrist_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, wrist):
        """
        x: [B, T, C=6]  (time-major in dim1)
        wrist: [B] or [B,1] integer indices (0 or 1)
        """
        # 1. Permute the input tensor to fit cnn order
        # x: [batch, time(1024/2048), channels(6)] -> [batch, channels, time]
        x = x.permute(0, 2, 1)
        
        # 2. CNN: extract local motion features
        features = self.cnn(x) # [B, 256, T'], where T' is ~= T/8
        features = features.permute(0, 2, 1) # [B, T', 256] (for GRU)
        
        # 3. GRU: capture temporal dependencies
        gru_out, _ = self.gru(features) # [B, T', hidden_size*2]
        
        # 4. Attention-weighted pooling 
        # compute logits: [B, T', 1], then softmax over time dim (dim=1)
        attn_logits = self.attention(gru_out).squeeze(-1)  # -> [B, T']
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # -> [B, T', 1]
        attended = (gru_out * attn_weights).sum(dim=1)  # -> [B, hidden_size*2]
        
        # 5. Global pooling (capture overall patterns)
        pooled_mean = gru_out.mean(dim=1)  # [B, hidden_size*2]
        pooled_max = gru_out.max(dim=1)[0] # [B, hidden_size*2]

        # 6. Combine all the temporal features -> [B, hidden_size*6]
        time_features = torch.cat([
            attended,
            pooled_mean,
            pooled_max
        ], dim=1)
        
        # 7. Wrist embedding
        wrist_embed = self.wrist_embed(wrist.long()) # -> [B, wrist_embed_dim]
        
        # 8. Concat GRU output + wrist info
        combined = torch.cat([time_features, wrist_embed], dim=1) # [B, hidden_size*6 + wrist_embed_dim]
        
        # 9. Final classification
        out = self.classifier(combined) # [B, num_classes]
        
        return out