"""
VAE Synthetic Data Generation - Per Movement
=============================================

Step 2: Generate balanced synthetic data for each movement type
Uses pretrained VAE models to create augmented training data.
"""

import torch
import numpy as np
from pathlib import Path
from collections import Counter
from tremor_VAE import TremorVAE


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Configuration")
print("="*70)

config = {
    'checkpoint_dir': Path('checkpoints/vae_per_movement'),
    'output_dir': Path('data/synthetic_per_movement'),
    'samples_per_class': 1000,  # Generate 1000 samples per class per movement
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'class_names': {0: 'Healthy', 1: 'Parkinson'},  # Binary (no 'Other')
}

config['output_dir'].mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  • Device: {config['device']}")
print(f"  • Checkpoint dir: {config['checkpoint_dir']}")
print(f"  • Output dir: {config['output_dir']}")
print(f"  • Samples per class: {config['samples_per_class']}")
print(f"  • Classes: {list(config['class_names'].values())}")


# ============================================================================
# 2. LOAD TRAINING RESULTS & DISCOVER MOVEMENTS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Loading Training Results")
print("="*70)

results_path = config['checkpoint_dir'] / 'training_results.pt'

if not results_path.exists():
    raise FileNotFoundError(
        f"Training results not found at {results_path}\n"
        "Please run '1_vae_pretrain_per_movement.py' first!"
    )

training_results = torch.load(results_path,  weights_only=False)

print(f"✓ Found {len(training_results)} trained VAE models:")
for model_name, results in training_results.items():
    movement = model_name.replace("_vae", "")
    val_loss = results['best_val_loss']
    print(f"  • {movement:<25s} (val_loss: {val_loss:.4f})")


# ============================================================================
# 3. LOAD VAE MODELS
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Loading VAE Models")
print("="*70)

vae_models = {}

for model_name, results in training_results.items():
    checkpoint_path = results['checkpoint_path']
    
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}, skipping...")
        continue
    
    # Create VAE
    vae = TremorVAE(
        latent_dim=256,
        signal_length=1024,
        dropout=0.3
    ).to(config['device'])
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=config['device'], weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    vae_models[model_name] = vae
    
    movement = model_name.replace("_vae", "")
    print(f"  ✓ {movement:<25s} loaded")

print(f"\n✓ Loaded {len(vae_models)} VAE models successfully")


# ============================================================================
# 4. GENERATE SYNTHETIC DATA FUNCTION
# ============================================================================

def generate_balanced_data(
    vae, movement_name, movement_idx,
    samples_per_class, class_names, device
):
    """
    Generate fully formatted synthetic data matching the real dataset.
    Returns dict with tuple samples:
        (
            signal_tensor (2,1024,6)
            handedness_tensor (scalar long)
            movement_tensor  (scalar long)
            label_tensor     (scalar long)
            metadata_tensor  (8,)
        )
    """

    vae.eval()
    print(f"\n  Generating data for {movement_name}...")

    samples = []

    with torch.no_grad():
        for label in class_names.keys():

            for _ in range(samples_per_class):

                # ------------------------------
                # 1. Label tensor
                # ------------------------------
                label_tensor = torch.tensor(label, dtype=torch.long)

                # ------------------------------
                # 2. Generate signal + VAE metadata
                # ------------------------------
                signal, raw_meta = vae.generate(
                    1, label_tensor.unsqueeze(0), device=device
                )
                signal = signal.squeeze(0).cpu().float()     # (2,1024,6)
                raw_meta = raw_meta.squeeze(0).cpu().float() # not used

                # ------------------------------
                # 3. Synthetic handedness (0/1)
                # ------------------------------
                handedness_tensor = torch.randint(
                    low=0, high=2, size=(1,), dtype=torch.long
                ).squeeze(0)

                # ------------------------------
                # 4. Movement index scalar
                # ------------------------------
                movement_tensor = torch.tensor(
                    movement_idx, dtype=torch.long
                )

                # ------------------------------
                # 5. Metadata vector (8 features)
                # ------------------------------
                # Replace this block with REAL metadata sampling if desired.
                metadata_tensor = torch.randn(8, dtype=torch.float32)

                # ------------------------------
                # 6. Create final tuple matching dataset spec
                # ------------------------------
                sample_tuple = (
                    signal,            # (2,1024,6)
                    handedness_tensor, # scalar
                    movement_tensor,   # scalar
                    label_tensor,      # scalar
                    metadata_tensor    # (8,)
                )

                samples.append(sample_tuple)

            print(f"    • {class_names[label]:<15s}: {samples_per_class:,} samples")

    return {
        'movement_name': movement_name,
        'movement_idx': movement_idx,
        'samples': samples,
        'num_samples': len(samples)
    }


# ============================================================================
# 5. GENERATE SYNTHETIC DATA FOR ALL MOVEMENTS
# ============================================================================
movement_names = [m.replace("_vae", "") for m in vae_models.keys()]
movement_to_idx = {name: idx for idx, name in enumerate(movement_names)}

all_synthetic_data = {}  # reset

print("\n" + "="*70)
print("STEP 4: Generating Synthetic Data")
print("="*70)

for model_name, vae in vae_models.items():
    movement_name = model_name.replace("_vae", "")
    movement_idx = movement_to_idx[movement_name]

    synthetic_data = generate_balanced_data(
        vae=vae,
        movement_name=movement_name,
        movement_idx=movement_idx,
        samples_per_class=config['samples_per_class'],
        class_names=config['class_names'],
        device=config['device']
    )

    save_path = config['output_dir'] / f"{movement_name}_synthetic.pt"
    torch.save(synthetic_data, save_path)

    all_synthetic_data[movement_name] = synthetic_data

    print(f"  ✓ Saved: {save_path}")
    print(f"    Samples: {synthetic_data['num_samples']:,}")


# ============================================================================
# 6. CREATE COMBINED DATASET (All Movements Together)
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Creating Combined Dataset")
print("="*70)

combined_samples = []

for movement_name, movement_data in all_synthetic_data.items():
    n_samples = movement_data['num_samples']
    print(f"  • {movement_name:<25s}: {n_samples:,} samples")
    combined_samples.extend(movement_data['samples'])

combined_dataset = {
    'samples': combined_samples,
    'movement_to_idx': movement_to_idx,
    'idx_to_movement': {i: n for n, i in movement_to_idx.items()},
    'total_samples': len(combined_samples),
}

combined_path = config['output_dir'] / "all_movements_synthetic.pt"
torch.save(combined_dataset, combined_path)

print(f"\n✓ Combined dataset saved:")
print(f"  → {combined_path}")
print(f"  Total samples: {len(combined_samples):,}")


# ============================================================================
# 7. STATISTICS & SUMMARY
# ============================================================================
from collections import Counter

print("\n" + "="*70)
print("GENERATION STATISTICS")
print("="*70)

labels_all = [sample[3].item() for sample in combined_dataset['samples']]
label_counts = Counter(labels_all)

print(f"\nClass Distribution:")
for label, name in config['class_names'].items():
    count = label_counts[label]
    pct = 100 * count / len(labels_all)
    print(f"{name:<15s} {count:>10,} {pct:>10.1f}%")

print(f"\nTotal samples: {len(labels_all):,}")

# ============================================================================
# 8. FINAL MESSAGE
# ============================================================================
print("\n" + "="*70)
print("SYNTHETIC DATA GENERATION COMPLETE!")
print("="*70)

print(f"\n📁 Generated Files:")
print(f"  • Per-movement datasets: {len(all_synthetic_data)} files")
print(f"  • Combined dataset: all_movements_synthetic.pt")
print(f"  • Generation summary: generation_summary.pt")

print(f"\n✅ Ready for fine-tuning!")
print(f"   Use 'all_movements_synthetic.pt' for pretraining TremorNetV9")
print(f"   Then fine-tune on real data for best results")

print("\n" + "="*70)
print("Next Step: Fine-tune TremorNetV9 with synthetic + real data")
print("="*70)