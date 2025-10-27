from torch import nn, cat

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
        self.cnn = nn.Sequential(
            # 1.1 Conv layer 1
            nn.Conv1d(
                in_channels= 6, # 6 IMU channels
                out_channels= 64,
                kernel_size= 5, stride=2, padding=2), # T -> T/2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 1.1 Conv layer 2
            nn.Conv1d(
                in_channels= 64,
                out_channels= 128,
                kernel_size= 3, stride=2, padding=1), # T/2 -> T/4
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Dropout(dropout)
        )
        
        # 2. GRU for temporal modeling (time-series data)
        # -------------------------------------------------
        self.gru = nn.GRU(
            input_size= 128, # input from CNN: 128 features per time step (after downsampling)
            hidden_size= hidden_size,
            num_layers=2, # stacking 2 GRU together
            batch_first= True,
            bidirectional= True, # outputs features in both time directions -> doubles hidden size
            dropout= dropout # applied between GRU layers (not after the last one)
        )
        
        # 3. Embedding wrist (left/ right)
        # ---------------------------------
        self.wrist_embed = nn.Embedding(
            num_embeddings= 2, # 0 = Left, 1 = Right
            embedding_dim= wrist_embed_dim 
            # each wrist type is represented by a learned 'wrist_embed_dim'-dimensional vector instead of a single scalar.
        ) # Output shape -> [B, wrist_embed_dim]
        
        # 4. Classifier
        # ---------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2 + wrist_embed_dim, 128), # concat(GRU output, wrist embedding)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, wrist):
        # 1. Permute the input tensor to fit cnn order
        # x: [batch, time(1024/2048), channels(6)] -> [batch, channels, time]
        x = x.permute(0, 2, 1)
        
        # 2. CNN: extract local motion features
        features = self.cnn(x) # [B, 128, T/4]
        features = features.permute(0, 2, 1) # [B, T/4, 128] (for GRU)
        
        # 3. GRU: capture temporal dependencies
        gru_out, _ = self.gru(features)
        last_out = gru_out[:, -1, :] # take the last timestep output -> [B, hidden_size * 2]
        
        # 4. Wrist embedding
        wrist_embed = self.wrist_embed(wrist.long()) # -> [B, wrist_embed_dim]
        
        # 5. Concat GRU output + wrist info
        combined = cat([last_out, wrist_embed], dim=1)
        
        # 6. Final calssification
        out = self.classifier(combined)
        
        return out