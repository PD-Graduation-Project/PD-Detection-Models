from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN, ModelParameters
import numpy as np
import pickle
import os

# Configuration
HIDDEN_DIM = 24  # Make sure this is defined
GAMMA = 1  # Make sure this is defined
OUTPUT_DIR = "./synthetic_data/"  # Make sure this is defined

def train_timegan_model(movement_name, movement_data, n_epochs=50):
    """
    Train a TimeGAN model for a specific movement.
    
    Args:
        movement_name: Name of the movement
        movement_data: Training data with shape (n_sequences, seq_len, n_features)
        n_epochs: Number of training epochs
    
    Returns:
        synth: Trained TimeGAN model
        model_path: Path where the model was saved
    """
    print(f"\n--- Training TimeGAN for Movement: {movement_name} ---")
    
    # Ensure data shape is correct
    n_sequences, seq_len, n_features = movement_data.shape
    print(f"Data shape: n_sequences={n_sequences}, seq_len={seq_len}, n_features={n_features}")
    
    # Define model parameters
    model_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        latent_dim=HIDDEN_DIM,
        gamma=GAMMA
    )
    
    # Initialize TimeGAN
    synth = TimeGAN(model_parameters=model_args)
    
    # Manually set the dimensions before training
    synth.seq_len = seq_len
    synth.n_seq = n_features  # n_seq refers to number of features/columns
    synth.num_cols = n_features
    
    # Train the model
    print(f"Starting training for {n_epochs} epochs...")
    synth.train(movement_data, train_steps=n_epochs)
    
    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, f"{movement_name}_timegan_model.pkl")
    synth.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return synth, model_path


def generate_synthetic_data(movement_name, n_samples=100, synth=None, model_path=None):
    """
    Generate synthetic data using a trained TimeGAN model.
    Can either use a provided model or load from a saved model file.
    
    Args:
        movement_name: Name of the movement
        n_samples: Number of synthetic sequences to generate
        synth: Trained TimeGAN model (optional, if not provided will load from file)
        model_path: Path to saved model file (optional, will use default path if not provided)
    
    Returns:
        synth_data: Generated synthetic data
        data_path: Path where the synthetic data was saved
    """
    print(f"\n--- Generating Synthetic Data for Movement: {movement_name} ---")
    
    # Load model if not provided
    if synth is None:
        if model_path is None:
            model_path = os.path.join(OUTPUT_DIR, f"{movement_name}_timegan_model.pkl")
        
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        synth = TimeGAN.load(model_path)
        print(f"✓ Model loaded successfully")
    else:
        print(f"Using provided model")
    
    # Generate synthetic data
    synth_data = synth.sample(n_samples)
    print(f"Generated synthetic data shape: {synth_data.shape}")
    
    # Save the synthetic data
    data_path = os.path.join(OUTPUT_DIR, f"{movement_name}_synthetic_data.npy")
    np.save(data_path, synth_data)
    print(f"✓ Synthetic data saved to: {data_path}")
    
    return synth_data, data_path

