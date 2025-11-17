import pandas as pd
import numpy as np
from pathlib import Path
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from tqdm import tqdm


class PADSDataSynthesizer:
    """
    Complete synthesizer for Parkinson's Disease Smartwatch Dataset.
    Generates synthetic metadata AND signal data (left + right wrist IMU).
    """
    
    def __init__(self, preprocessed_dir: Path):
        """
        Initialize synthesizer with preprocessed dataset directory.
        
        Parameters
        ----------
        preprocessed_dir : Path
            Path to the root directory containing preprocessed movement folders
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.metadata_synthesizer = None
        self.signal_synthesizer = None
        self.metadata_schema = None
        self.signal_schema = None
        self.real_metadata = None
        self.real_signals = None
        
    def load_data_from_npz(self):
        """
        Load all data (metadata + signals) from .npz files.
        
        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            (metadata_df, signals_df)
        """
        metadata_records = []
        signal_records = []
        
        # Iterate through all movement types
        for movement_dir in self.preprocessed_dir.iterdir():
            if not movement_dir.is_dir():
                continue
                
            movement_name = movement_dir.name
            
            # Iterate through label subdirectories
            for label_dir in movement_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                    
                label_name = label_dir.name
                label_map = {"Healthy": 0, "Parkinson": 1, "Other": 2}
                label = label_map.get(label_name, -1)
                
                # Load each subject's .npz file
                for npz_file in label_dir.glob("*.npz"):
                    data = np.load(npz_file, allow_pickle=True)
                    
                    # Get signals (tuple of left, right)
                    signals = data['signal']
                    left_signal = signals[0]  # (1024, 6)
                    right_signal = signals[1]  # (1024, 6)
                    
                    # Create unique row_id
                    row_id = f"{int(data['subject_id'])}_{movement_name}"
                    
                    # Metadata record
                    metadata_record = {
                        'row_id': row_id,
                        'subject_id': int(data['subject_id']),
                        'movement': movement_name,
                        'label': int(data['label']),
                        'wrist': int(data['wrist']),
                        **data['metadata'].item()
                    }
                    metadata_records.append(metadata_record)
                    
                    # Signal record - flatten both signals
                    # Left: 1024x6 = 6144 values, Right: 1024x6 = 6144 values
                    signal_record = {
                        'row_id': row_id,
                        'label': int(data['label']),
                        'movement': movement_name,
                    }
                    
                    # Add flattened left signal features
                    left_flat = left_signal.flatten()
                    for i in range(len(left_flat)):
                        signal_record[f'left_{i}'] = left_flat[i]
                    
                    # Add flattened right signal features
                    right_flat = right_signal.flatten()
                    for i in range(len(right_flat)):
                        signal_record[f'right_{i}'] = right_flat[i]
                    
                    signal_records.append(signal_record)
        
        metadata_df = pd.DataFrame(metadata_records)
        signals_df = pd.DataFrame(signal_records)
        
        print(f"Loaded {len(metadata_df)} records from {len(metadata_df['subject_id'].unique())} subjects")
        print(f"Metadata shape: {metadata_df.shape}")
        print(f"Signals shape: {signals_df.shape}")
        
        return metadata_df, signals_df
    
    def prepare_metadata_schema(self, df):
        """Define metadata schema for SDV."""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Set column types using the correct API
        metadata.update_column('row_id', sdtype='id')
        metadata.update_column('subject_id', sdtype='numerical')
        metadata.update_column('movement', sdtype='categorical')
        metadata.update_column('label', sdtype='categorical')
        metadata.update_column('wrist', sdtype='categorical')
        metadata.update_column('gender', sdtype='categorical')
        metadata.update_column('appearance_in_kinship', sdtype='categorical')
        metadata.update_column('appearance_in_first_grade_kinship', sdtype='categorical')
        metadata.update_column('effect_of_alcohol_on_tremor', sdtype='categorical')
        
        metadata.set_primary_key('row_id')
        
        return metadata
    
    def prepare_signal_schema(self, df):
        """Define signal schema for SDV."""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Set column types using the correct API
        metadata.update_column('row_id', sdtype='id')
        metadata.update_column('label', sdtype='categorical')
        metadata.update_column('movement', sdtype='categorical')
        
        # All signal columns are numerical (already detected correctly)
        metadata.set_primary_key('row_id')
        
        return metadata
    
    def train(self, epochs=300, batch_size=500, compress_dims=(128, 128), decompress_dims=(128, 128)):
        """
        Train TVAE synthesizers for both metadata and signals.
        
        Parameters
        ----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        compress_dims : tuple
            Encoder dimensions for TVAE
        decompress_dims : tuple
            Decoder dimensions for TVAE
        """
        print("Loading data from dataset...")
        self.real_metadata, self.real_signals = self.load_data_from_npz()
        
        # Train metadata synthesizer
        print("\n" + "="*60)
        print("TRAINING METADATA SYNTHESIZER")
        print("="*60)
        self.metadata_schema = self.prepare_metadata_schema(self.real_metadata)
        
        self.metadata_synthesizer = TVAESynthesizer(
            metadata=self.metadata_schema,
            epochs=epochs,
            batch_size=batch_size,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            enforce_min_max_values=True,
            enforce_rounding=True
        )
        
        print(f"Training metadata TVAE (epochs={epochs})...")
        self.metadata_synthesizer.fit(self.real_metadata)
        print("✓ Metadata synthesizer trained!")
        
        # Train signal synthesizer
        print("\n" + "="*60)
        print("TRAINING SIGNAL SYNTHESIZER")
        print("="*60)
        self.signal_schema = self.prepare_signal_schema(self.real_signals)
        
        self.signal_synthesizer = TVAESynthesizer(
            metadata=self.signal_schema,
            epochs=epochs,
            batch_size=batch_size,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            enforce_min_max_values=False,  # Allow full range for signals
            enforce_rounding=False
        )
        
        print(f"Training signal TVAE (epochs={epochs})...")
        self.signal_synthesizer.fit(self.real_signals)
        print("✓ Signal synthesizer trained!")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
    
    def generate_synthetic_data(self, num_samples=100):
        """
        Generate complete synthetic dataset (metadata + signals).
        
        Parameters
        ----------
        num_samples : int
            Number of synthetic samples to generate
            
        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            (synthetic_metadata, synthetic_signals)
        """
        if self.metadata_synthesizer is None or self.signal_synthesizer is None:
            raise ValueError("Synthesizers not trained. Call train() first.")
        
        print(f"\nGenerating {num_samples} synthetic samples...")
        
        # Generate metadata
        synthetic_metadata = self.metadata_synthesizer.sample(num_rows=num_samples)
        
        # Generate signals with matching row_ids and labels
        synthetic_signals_list = []
        
        for _, row in tqdm(synthetic_metadata.iterrows(), total=len(synthetic_metadata), desc="Generating signals"):
            # Sample one signal conditioned on label and movement
            condition = {
                'label': row['label'],
                'movement': row['movement']
            }
            
            try:
                signal_sample = self.signal_synthesizer.sample_from_conditions(
                    conditions=[condition],
                    max_tries_per_batch=100
                )
                # Update row_id to match metadata
                signal_sample['row_id'] = row['row_id']
                synthetic_signals_list.append(signal_sample)
            except Exception as e:
                # Fallback: sample without conditions
                signal_sample = self.signal_synthesizer.sample(num_rows=1)
                signal_sample['row_id'] = row['row_id']
                signal_sample['label'] = row['label']
                signal_sample['movement'] = row['movement']
                synthetic_signals_list.append(signal_sample)
        
        synthetic_signals = pd.concat(synthetic_signals_list, ignore_index=True)
        
        print("✓ Generation complete!")
        return synthetic_metadata, synthetic_signals
    
    def save_as_npz(self, synthetic_metadata, synthetic_signals, output_dir="synthetic_data"):
        """
        Save synthetic data in the same .npz format as original dataset.
        
        Parameters
        ----------
        synthetic_metadata : pd.DataFrame
            Synthetic metadata
        synthetic_signals : pd.DataFrame
            Synthetic signals
        output_dir : str
            Directory to save synthetic .npz files
        """
        output_path = Path(output_dir)
        
        print(f"\nSaving synthetic data to {output_path}...")
        
        for _, meta_row in tqdm(synthetic_metadata.iterrows(), total=len(synthetic_metadata), desc="Saving files"):
            row_id = meta_row['row_id']
            
            # Get corresponding signal
            signal_row = synthetic_signals[synthetic_signals['row_id'] == row_id].iloc[0]
            
            # Reconstruct signals from flattened data
            left_cols = [col for col in signal_row.index if col.startswith('left_')]
            right_cols = [col for col in signal_row.index if col.startswith('right_')]
            
            left_signal = signal_row[left_cols].values.reshape(1024, 6)
            right_signal = signal_row[right_cols].values.reshape(1024, 6)
            
            # Prepare metadata dict
            metadata_dict = {
                'age_at_diagnosis': meta_row['age_at_diagnosis'],
                'age': meta_row['age'],
                'height': meta_row['height'],
                'weight': meta_row['weight'],
                'gender': meta_row['gender'],
                'appearance_in_kinship': meta_row['appearance_in_kinship'],
                'appearance_in_first_grade_kinship': meta_row['appearance_in_first_grade_kinship'],
                'effect_of_alcohol_on_tremor': meta_row['effect_of_alcohol_on_tremor'],
            }
            
            # Create directory structure
            label_name = {0: "Healthy", 1: "Parkinson", 2: "Other"}.get(meta_row['label'], "Unknown")
            save_dir = output_path / meta_row['movement'] / label_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as .npz
            np.savez_compressed(
                save_dir / f"{meta_row['subject_id']:.0f}.npz",
                signal=(left_signal.astype(np.float32), right_signal.astype(np.float32)),
                label=int(meta_row['label']),
                wrist=int(meta_row['wrist']),
                subject_id=int(meta_row['subject_id']),
                metadata=metadata_dict
            )
        
        print(f"✓ Saved {len(synthetic_metadata)} synthetic .npz files to {output_path.resolve()}")
    
    def save_models(self, metadata_path="tvae_metadata.pkl", signal_path="tvae_signals.pkl"):
        """Save both trained synthesizers."""
        if self.metadata_synthesizer:
            self.metadata_synthesizer.save(metadata_path)
            print(f"✓ Metadata model saved to {metadata_path}")
        
        if self.signal_synthesizer:
            self.signal_synthesizer.save(signal_path)
            print(f"✓ Signal model saved to {signal_path}")
    
    def load_models(self, metadata_path="tvae_metadata.pkl", signal_path="tvae_signals.pkl"):
        """Load both trained synthesizers."""
        self.metadata_synthesizer = TVAESynthesizer.load(metadata_path)
        self.signal_synthesizer = TVAESynthesizer.load(signal_path)
        print(f"✓ Models loaded from {metadata_path} and {signal_path}")


# Example usage
# -------------
if __name__ == "__main__":
    
    # 1. Initialize synthesizer
    preprocessed_dir = Path("../../../project_datasets/tremor/")
    synthesizer = PADSDataSynthesizer(preprocessed_dir)
    
    # 2. Train both synthesizers (metadata + signals)
    synthesizer.train(
        epochs=300,
        batch_size=500,
        compress_dims=(128, 128),
        decompress_dims=(128, 128)
    )
    
    # 3. Generate synthetic data
    synthetic_metadata, synthetic_signals = synthesizer.generate_synthetic_data(num_samples=200)
    
    # 4. Display sample
    print("\n" + "="*60)
    print("SYNTHETIC METADATA SAMPLE")
    print("="*60)
    print(synthetic_metadata.head())
    
    print("\n" + "="*60)
    print("SYNTHETIC SIGNALS SAMPLE")
    print("="*60)
    print(synthetic_signals.iloc[:, :10].head())  # Show first 10 signal columns
    
    # 5. Save as .npz files (same format as original dataset)
    synthesizer.save_as_npz(
        synthetic_metadata,
        synthetic_signals,
        output_dir="synthetic_pads_dataset"
    )
    
    # 6. Save models for reuse
    synthesizer.save_models(
        metadata_path="pads_tvae_metadata.pkl",
        signal_path="pads_tvae_signals.pkl"
    )
    
    # 7. Optional: Save as CSV for inspection
    synthetic_metadata.to_csv("synthetic_metadata.csv", index=False)
    print("\n✓ Synthetic metadata saved to 'synthetic_metadata.csv'")
    
    print("\n" + "="*60)
    print("SYNTHESIS COMPLETE!")
    print("="*60)
    print(f"Generated {len(synthetic_metadata)} complete synthetic samples")
    print(f"Each sample contains:")
    print(f"  - Left wrist signal: (1024, 6)")
    print(f"  - Right wrist signal: (1024, 6)")
    print(f"  - Complete metadata with clinical features")
    print(f"\nFiles saved in the same .npz format as original dataset!")