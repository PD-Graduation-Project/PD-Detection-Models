"""
VAE Pretraining - Per Movement
================================

Step 1: Train a VAE for each movement type separately
This learns movement-specific tremor patterns and generates better synthetic data.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from data.dataloader import create_tremor_dataloaders
from tremor_VAE import TremorVAE, vae_loss_function


# ============================================================================
# 1. LOAD DATA PER MOVEMENT
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Loading Movement-Specific Dataloaders")
print("="*70)

all_dataloaders = create_tremor_dataloaders(
    "../project_datasets/tremor/",
    batch_size=32,
    include_other=False,
    print_details=True,
    per_movement=True
)

print(f"\n✓ Loaded dataloaders for {len(all_dataloaders)} movements:")
for movement_name in all_dataloaders.keys():
    train_size = len(all_dataloaders[movement_name]["train"].dataset)
    val_size = len(all_dataloaders[movement_name]["val"].dataset)
    print(f"  • {movement_name:20s} → Train: {train_size:4d} | Val: {val_size:4d}")


# ============================================================================
# 2. CREATE VAE MODELS (one per movement)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Creating VAE Models")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

vae_models = {}

for movement_name in all_dataloaders.keys():
    model_name = f"{movement_name}_vae"
    
    vae = TremorVAE(
        latent_dim=256,
        signal_length=1024,
        dropout=0.3
    ).to(device)
    
    vae_models[model_name] = vae
    
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"  • {model_name:25s} → {num_params:,} parameters")

print(f"\n✓ Created {len(vae_models)} VAE models")


# ============================================================================
# 3. TRAINING CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Training Configuration")
print("="*70)

config = {
    'epochs': 100,
    'lr': 1e-3,
    'beta': 0.001,  # KL weight
    'metadata_weight': 0.1,
    'checkpoint_dir': Path('checkpoints/vae_per_movement'),
    'log_interval': 10,
}

config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  • Epochs: {config['epochs']}")
print(f"  • Learning Rate: {config['lr']}")
print(f"  • Beta (KL weight): {config['beta']}")
print(f"  • Metadata weight: {config['metadata_weight']}")
print(f"  • Checkpoint dir: {config['checkpoint_dir']}")


# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_vae_one_epoch(vae, train_loader, optimizer, beta, metadata_weight, device):
    """Train VAE for one epoch"""
    vae.train()
    
    epoch_loss = 0
    epoch_signal_loss = 0
    epoch_meta_loss = 0
    epoch_kl_loss = 0
    
    for batch in train_loader:
        signals, handedness, movements, labels, metadata = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        # Forward
        recon_signals, recon_metadata, mu, logvar = vae(signals, metadata, labels)
        
        # Loss
        loss, sig_loss, meta_loss, kl_loss = vae_loss_function(
            recon_signals, signals,
            recon_metadata, metadata,
            mu, logvar, 
            beta=beta,
            metadata_weight=metadata_weight
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_signal_loss += sig_loss.item()
        epoch_meta_loss += meta_loss.item()
        epoch_kl_loss += kl_loss.item()
    
    n_batches = len(train_loader)
    return (epoch_loss / n_batches, 
            epoch_signal_loss / n_batches,
            epoch_meta_loss / n_batches,
            epoch_kl_loss / n_batches)


def validate_vae(vae, val_loader, beta, metadata_weight, device):
    """Validate VAE"""
    vae.eval()
    
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            signals, handedness, movements, labels, metadata = [b.to(device) for b in batch]
            
            recon_signals, recon_metadata, mu, logvar = vae(signals, metadata, labels)
            
            loss, _, _, _ = vae_loss_function(
                recon_signals, signals,
                recon_metadata, metadata,
                mu, logvar,
                beta=beta,
                metadata_weight=metadata_weight
            )
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


# ============================================================================
# 5. TRAIN ALL VAE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Training VAE Models")
print("="*70)

training_results = {}

for model_name, vae in vae_models.items():
    # Extract movement name
    movement_name = model_name.replace("_vae", "")
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Get dataloaders
    train_loader = all_dataloaders[movement_name]["train"]
    val_loader = all_dataloaders[movement_name]["val"]
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(vae.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'signal_loss': [],
        'meta_loss': [],
        'kl_loss': []
    }
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, sig_loss, meta_loss, kl_loss = train_vae_one_epoch(
            vae, train_loader, optimizer,
            config['beta'], config['metadata_weight'], device
        )
        
        # Validate
        val_loss = validate_vae(
            vae, val_loader,
            config['beta'], config['metadata_weight'], device
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['signal_loss'].append(sig_loss)
        history['meta_loss'].append(meta_loss)
        history['kl_loss'].append(kl_loss)
        
        # Print progress
        if (epoch + 1) % config['log_interval'] == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train: {train_loss:.4f} "
                  f"(sig:{sig_loss:.4f}, meta:{meta_loss:.4f}, kl:{kl_loss:.4f}) | "
                  f"Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = config['checkpoint_dir'] / f"{model_name}.pth"
            torch.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': config
            }, save_path)
            
            if (epoch + 1) % config['log_interval'] == 0 or epoch == 0:
                print(f"  → Saved checkpoint (val_loss: {val_loss:.4f})")
        
        scheduler.step()
    
    # Store results
    training_results[model_name] = {
        'best_val_loss': best_val_loss,
        'history': history,
        'checkpoint_path': config['checkpoint_dir'] / f"{model_name}.pth"
    }
    
    print(f"\n✓ {model_name} training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Saved to: {training_results[model_name]['checkpoint_path']}")


# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

print(f"\n{'Movement':<25s} {'Best Val Loss':>15s} {'Checkpoint Path'}")
print("-" * 70)

for model_name, results in training_results.items():
    movement = model_name.replace("_vae", "")
    print(f"{movement:<25s} {results['best_val_loss']:>15.4f}  ✓")

print(f"\n✓ All {len(vae_models)} VAE models trained successfully!")
print(f"✓ Checkpoints saved to: {config['checkpoint_dir']}")

# Save training results
results_path = config['checkpoint_dir'] / 'training_results.pt'
torch.save(training_results, results_path)
print(f"✓ Training results saved to: {results_path}")

print("\n" + "="*70)
print("Next Step: Run '2_vae_generate_synthetic_data.py'")
print("="*70)