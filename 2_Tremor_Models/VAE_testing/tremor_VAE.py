"""
Tabular VAE for Tremor Signal Generation
=========================================

A Variational Autoencoder designed to:
1. Learn compressed representations of tremor signals + metadata
2. Generate synthetic balanced data for pretraining
3. Provide pretrained encoder weights for TremorNetV9

Architecture:
- Encoder: Signal (2, T, 6) + Metadata (8) -> Latent (256)
- Decoder: Latent (256) -> Signal (2, T, 6) + Metadata (8)
- Conditional generation by class label (Healthy/PD/Other)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TremorEncoder(nn.Module):
    """
    Encodes tremor signals + metadata into latent space.
    Similar architecture to TremorNetV9 for weight transfer.
    """
    def __init__(self, latent_dim=256, dropout=0.3):
        super().__init__()
        
        # === SIGNAL ENCODER (per wrist) ===
        # Multi-scale CNN
        self.conv_fast = nn.Conv1d(6, 64, kernel_size=3, stride=2, padding=1)
        self.conv_mid = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv_slow = nn.Conv1d(6, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(192)
        
        self.conv2 = nn.Conv1d(192, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # === METADATA ENCODER ===
        self.metadata_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 64)
        )
        
        # === LABEL EMBEDDING (conditional) ===
        self.label_embed = nn.Embedding(3, 32)  # Healthy/PD/Other
        
        # === FUSION -> LATENT ===
        # Signal features: 256 (left) + 256 (right) + 64 (metadata) + 32 (label) = 608
        self.fusion = nn.Sequential(
            nn.Linear(608, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.GELU()
        )
        
        # VAE: mean and log_variance
        self.fc_mu = nn.Linear(384, latent_dim)
        self.fc_logvar = nn.Linear(384, latent_dim)
        
    def _encode_signal(self, x):
        """x: [B, 6, T] -> [B, 256]"""
        # Multi-scale
        fast = self.conv_fast(x)
        mid = self.conv_mid(x)
        slow = self.conv_slow(x)
        
        x = torch.cat([fast, mid, slow], dim=1)
        x = F.gelu(self.bn1(x))
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)  # [B, 256]
        return x
    
    def forward(self, signals, metadata, labels):
        """
        Args:
            signals: [B, 2, T, 6] - left and right wrist
            metadata: [B, 8]
            labels: [B] - class labels (0=Healthy, 1=PD, 2=Other)
        Returns:
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        # Encode each wrist separately
        left_signal = signals[:, 0].permute(0, 2, 1)   # [B, 6, T]
        right_signal = signals[:, 1].permute(0, 2, 1)  # [B, 6, T]
        
        left_feat = self._encode_signal(left_signal)   # [B, 256]
        right_feat = self._encode_signal(right_signal) # [B, 256]
        
        # Encode metadata
        meta_feat = self.metadata_encoder(metadata)    # [B, 64]
        
        # Embed labels (conditional)
        label_feat = self.label_embed(labels.long())   # [B, 32]
        
        # Fuse all features
        combined = torch.cat([left_feat, right_feat, meta_feat, label_feat], dim=1)  # [B, 608]
        h = self.fusion(combined)  # [B, 384]
        
        # VAE parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class TremorDecoder(nn.Module):
    """
    Decodes latent vector back to signals + metadata.
    """
    def __init__(self, latent_dim=256, signal_length=1024, dropout=0.3):
        super().__init__()
        
        self.signal_length = signal_length
        
        # === LABEL EMBEDDING (conditional) ===
        self.label_embed = nn.Embedding(3, 32)
        
        # === LATENT -> FEATURES ===
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim + 32, 384),  # +32 for label
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 512),
            nn.GELU()
        )
        
        # === SIGNAL DECODER (per wrist) ===
        # Start with small temporal dimension and upsample
        self.signal_init = nn.Linear(256, 256 * 32)  # [B, 256] -> [B, 256, 32]
        
        # Transpose convolutions for upsampling
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)   # 64 -> 128
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)    # 128 -> 256
        self.deconv4 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)    # 256 -> 512
        self.deconv5 = nn.ConvTranspose1d(16, 6, kernel_size=4, stride=2, padding=1)     # 512 -> 1024
        
        self.bn_dec1 = nn.BatchNorm1d(128)
        self.bn_dec2 = nn.BatchNorm1d(64)
        self.bn_dec3 = nn.BatchNorm1d(32)
        self.bn_dec4 = nn.BatchNorm1d(16)
        
        # === METADATA DECODER ===
        self.metadata_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 8)  # Reconstruct all 8 metadata features
        )
        
    def _decode_signal(self, z):
        """z: [B, 256] -> [B, 6, T]"""
        # Initial projection
        x = self.signal_init(z)  # [B, 256*32]
        x = x.view(-1, 256, 32)  # [B, 256, 32]
        
        # Upsample
        x = F.gelu(self.bn_dec1(self.deconv1(x)))
        x = F.gelu(self.bn_dec2(self.deconv2(x)))
        x = F.gelu(self.bn_dec3(self.deconv3(x)))
        x = F.gelu(self.bn_dec4(self.deconv4(x)))
        x = self.deconv5(x)  # [B, 6, 1024]
        
        # Ensure exact length
        if x.shape[2] != self.signal_length:
            x = F.interpolate(x, size=self.signal_length, mode='linear', align_corners=False)
        
        return x  # [B, 6, T]
    
    def forward(self, z, labels):
        """
        Args:
            z: [B, latent_dim] - latent vector
            labels: [B] - class labels for conditional generation
        Returns:
            left_signal: [B, 6, T]
            right_signal: [B, 6, T]
            metadata: [B, 8]
        """
        # Embed labels
        label_feat = self.label_embed(labels.long())  # [B, 32]
        
        # Combine latent + label
        z_cond = torch.cat([z, label_feat], dim=1)  # [B, latent_dim + 32]
        
        # Project to feature space
        h = self.latent_proj(z_cond)  # [B, 512]
        
        # Split for left/right signals and metadata
        left_feat = h[:, :256]   # [B, 256]
        right_feat = h[:, 256:]  # [B, 256]
        
        # Decode signals
        left_signal = self._decode_signal(left_feat)   # [B, 6, T]
        right_signal = self._decode_signal(right_feat) # [B, 6, T]
        
        # Decode metadata (use average of left/right features)
        meta_feat = (left_feat + right_feat) / 2
        metadata = self.metadata_decoder(meta_feat)  # [B, 8]
        
        return left_signal, right_signal, metadata


class TremorVAE(nn.Module):
    """
    Complete VAE for tremor signal generation.
    """
    def __init__(self, latent_dim=256, signal_length=1024, dropout=0.3):
        super().__init__()
        
        self.encoder = TremorEncoder(latent_dim=latent_dim, dropout=dropout)
        self.decoder = TremorDecoder(latent_dim=latent_dim, signal_length=signal_length, dropout=dropout)
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, signals, metadata, labels):
        """
        Args:
            signals: [B, 2, T, 6]
            metadata: [B, 8]
            labels: [B]
        Returns:
            recon_signals: [B, 2, T, 6]
            recon_metadata: [B, 8]
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(signals, metadata, labels)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        left_recon, right_recon, meta_recon = self.decoder(z, labels)
        
        # Stack left/right
        recon_signals = torch.stack([left_recon, right_recon], dim=1).permute(0, 1, 3, 2)  # [B, 2, T, 6]
        
        return recon_signals, meta_recon, mu, logvar
    
    def generate(self, num_samples, labels, device='cuda'):
        """
        Generate synthetic samples.
        
        Args:
            num_samples: int
            labels: [num_samples] - desired class labels
            device: torch device
        Returns:
            signals: [num_samples, 2, T, 6]
            metadata: [num_samples, 8]
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=device)
            labels = labels.to(device)
            
            # Decode
            left_signal, right_signal, metadata = self.decoder(z, labels)
            
            # Stack
            signals = torch.stack([left_signal, right_signal], dim=1).permute(0, 1, 3, 2)
            
        return signals, metadata


def vae_loss_function(recon_signals, original_signals, 
                     recon_metadata, original_metadata,
                     mu, logvar, 
                     beta=0.001, metadata_weight=0.1):
    """
    VAE loss = Reconstruction Loss + KL Divergence
    
    Args:
        recon_signals: [B, 2, T, 6]
        original_signals: [B, 2, T, 6]
        recon_metadata: [B, 8]
        original_metadata: [B, 8]
        mu, logvar: [B, latent_dim]
        beta: KL weight (use small value for beta-VAE)
        metadata_weight: weight for metadata reconstruction
    """
    B = recon_signals.shape[0]
    
    # 1. Signal reconstruction loss (MSE)
    signal_recon_loss = F.mse_loss(recon_signals, original_signals, reduction='sum') / B
    
    # 2. Metadata reconstruction loss (MSE, ignore missing values)
    # Mask for valid metadata (not -1)
    valid_mask = (original_metadata != -1).float()
    metadata_recon_loss = (F.mse_loss(recon_metadata, original_metadata, reduction='none') * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    
    # 3. KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    # Total loss
    total_loss = signal_recon_loss + metadata_weight * metadata_recon_loss + beta * kl_loss
    
    return total_loss, signal_recon_loss, metadata_recon_loss, kl_loss


# ============================================================================
# PRETRAINING UTILITIES
# ============================================================================

def pretrain_vae(vae, train_loader, val_loader, 
                epochs=100, lr=1e-3, beta=0.001, 
                device='cuda', save_path='tremor_vae.pth'):
    """
    Pretrain the VAE on tremor data.
    
    Args:
        vae: TremorVAE model
        train_loader: DataLoader with (signals, handedness, movements, labels, metadata)
        val_loader: validation DataLoader
        epochs: training epochs
        lr: learning rate
        beta: KL weight
        device: torch device
        save_path: where to save best model
    """
    vae = vae.to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # === TRAIN ===
        vae.train()
        train_loss = 0
        train_signal_loss = 0
        train_meta_loss = 0
        train_kl_loss = 0
        
        for batch in train_loader:
            signals, handedness, movements, labels, metadata = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward
            recon_signals, recon_metadata, mu, logvar = vae(signals, metadata, labels)
            
            # Loss
            loss, sig_loss, meta_loss, kl_loss = vae_loss_function(
                recon_signals, signals,
                recon_metadata, metadata,
                mu, logvar, beta=beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_signal_loss += sig_loss.item()
            train_meta_loss += meta_loss.item()
            train_kl_loss += kl_loss.item()
        
        # === VALIDATE ===
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                signals, handedness, movements, labels, metadata = [b.to(device) for b in batch]
                recon_signals, recon_metadata, mu, logvar = vae(signals, metadata, labels)
                loss, _, _, _ = vae_loss_function(
                    recon_signals, signals,
                    recon_metadata, metadata,
                    mu, logvar, beta=beta
                )
                val_loss += loss.item()
        
        # Stats
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_signal_loss /= len(train_loader)
        train_meta_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train: {train_loss:.4f} (sig:{train_signal_loss:.4f}, meta:{train_meta_loss:.4f}, kl:{train_kl_loss:.4f}) | "
              f"Val: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), save_path)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        scheduler.step()
    
    print(f"\nPretraining complete! Best val loss: {best_val_loss:.4f}")
    return vae


def generate_balanced_dataset(vae, num_per_class=1000, device='cuda'):
    """
    Generate balanced synthetic dataset.
    
    Args:
        vae: trained TremorVAE
        num_per_class: samples to generate per class
        device: torch device
    Returns:
        synthetic_data: dict with 'signals', 'metadata', 'labels'
    """
    vae.eval()
    
    all_signals = []
    all_metadata = []
    all_labels = []
    
    for label in [0, 1, 2]:  # Healthy, PD, Other
        labels = torch.full((num_per_class,), label, dtype=torch.long, device=device)
        signals, metadata = vae.generate(num_per_class, labels, device=device)
        
        all_signals.append(signals.cpu())
        all_metadata.append(metadata.cpu())
        all_labels.append(labels.cpu())
    
    synthetic_data = {
        'signals': torch.cat(all_signals, dim=0),
        'metadata': torch.cat(all_metadata, dim=0),
        'labels': torch.cat(all_labels, dim=0)
    }
    
    print(f"Generated {len(synthetic_data['labels'])} balanced samples:")
    print(f"  Healthy: {(synthetic_data['labels']==0).sum()}")
    print(f"  PD: {(synthetic_data['labels']==1).sum()}")
    print(f"  Other: {(synthetic_data['labels']==2).sum()}")
    
    return synthetic_data


def transfer_encoder_weights(vae, tremor_net):
    """
    Transfer pretrained encoder weights to TremorNetV9.
    
    Args:
        vae: pretrained TremorVAE
        tremor_net: TremorNetV9 model
    """
    # Transfer conv layers
    tremor_net.conv_fast.load_state_dict(vae.encoder.conv_fast.state_dict())
    tremor_net.conv_mid.load_state_dict(vae.encoder.conv_mid.state_dict())
    tremor_net.conv_slow.load_state_dict(vae.encoder.conv_slow.state_dict())
    tremor_net.bn1.load_state_dict(vae.encoder.bn1.state_dict())
    tremor_net.conv2.load_state_dict(vae.encoder.conv2.state_dict())
    tremor_net.bn2.load_state_dict(vae.encoder.bn2.state_dict())
    
    print("✓ Transferred encoder weights from VAE to TremorNetV9")