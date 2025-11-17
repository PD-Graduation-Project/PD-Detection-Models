import pandas as pd
import numpy as np
from pathlib import Path
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata


class PADSDataSynthesizer:
    """
    Synthesizer for Parkinson's Disease Smartwatch Dataset using Tabular VAE.
    Generates synthetic tabular metadata that matches the distribution of real data.
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
        self.synthesizer = None
        self.metadata_schema = None
        self.real_data = None
        
    def load_metadata_from_npz(self):
        """
        Load all metadata from .npz files into a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all metadata fields from the dataset
        """
        records = []
        
        # Iterate through all movement types
        for movement_dir in self.preprocessed_dir.iterdir():
            if not movement_dir.is_dir():
                continue
                
            movement_name = movement_dir.name
            
            # Iterate through label subdirectories (Healthy, Parkinson, Other)
            for label_dir in movement_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                    
                label_name = label_dir.name
                label_map = {"Healthy": 0, "Parkinson": 1, "Other": 2}
                label = label_map.get(label_name, -1)
                
                # Load each subject's .npz file
                for npz_file in label_dir.glob("*.npz"):
                    data = np.load(npz_file, allow_pickle=True)
                    
                    record = {
                        'subject_id': int(data['subject_id']),
                        'movement': movement_name,
                        'label': int(data['label']),
                        'wrist': int(data['wrist']),  # handedness
                        **data['metadata'].item()  # unpack metadata dict
                    }
                    records.append(record)
        
        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} records from {len(df['subject_id'].unique())} subjects")
        print(f"Movements: {df['movement'].unique()}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def prepare_metadata_schema(self, df):
        """
        Define the metadata schema for SDV.
        
        Parameters
        ----------
        df : pd.DataFrame
            The real data DataFrame
            
        Returns
        -------
        SingleTableMetadata
            Metadata schema for the synthesizer
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Explicitly set column types
        metadata.update_column(column_name='subject_id', sdtype='id')
        metadata.update_column(column_name='movement', sdtype='categorical')
        metadata.update_column(column_name='label', sdtype='categorical')
        metadata.update_column(column_name='wrist', sdtype='categorical')
        metadata.update_column(column_name='gender', sdtype='categorical')
        metadata.update_column(column_name='appearance_in_kinship', sdtype='categorical')
        metadata.update_column(column_name='appearance_in_first_grade_kinship', sdtype='categorical')
        metadata.update_column(column_name='effect_of_alcohol_on_tremor', sdtype='categorical')
        
        # Set primary key
        metadata.set_primary_key('subject_id')
        
        return metadata
    
    def train(self, epochs=300, batch_size=500):
        """
        Train the TVAE synthesizer on the real data.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs (default: 300)
        batch_size : int, optional
            Batch size for training (default: 500)
        """
        print("Loading metadata from dataset...")
        self.real_data = self.load_metadata_from_npz()
        
        print("\nPreparing metadata schema...")
        self.metadata_schema = self.prepare_metadata_schema(self.real_data)
        
        print("\nInitializing TVAE Synthesizer...")
        self.synthesizer = TVAESynthesizer(
            metadata=self.metadata_schema,
            epochs=epochs,
            batch_size=batch_size,
            enforce_min_max_values=True,
            enforce_rounding=True
        )
        
        print(f"\nTraining TVAE (epochs={epochs}, batch_size={batch_size})...")
        self.synthesizer.fit(self.real_data)
        print("Training complete!")
        
    def generate_synthetic_data(self, num_samples=100):
        """
        Generate synthetic metadata samples.
        
        Parameters
        ----------
        num_samples : int, optional
            Number of synthetic samples to generate (default: 100)
            
        Returns
        -------
        pd.DataFrame
            Synthetic metadata samples
        """
        if self.synthesizer is None:
            raise ValueError("Synthesizer not trained. Call train() first.")
        
        print(f"\nGenerating {num_samples} synthetic samples...")
        synthetic_data = self.synthesizer.sample(num_rows=num_samples)
        
        return synthetic_data
    
    def save_model(self, filepath="tvae_model.pkl"):
        """
        Save the trained synthesizer to disk.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save the model (default: "tvae_model.pkl")
        """
        if self.synthesizer is None:
            raise ValueError("No trained model to save.")
        
        self.synthesizer.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="tvae_model.pkl"):
        """
        Load a trained synthesizer from disk.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the saved model (default: "tvae_model.pkl")
        """
        self.synthesizer = TVAESynthesizer.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def evaluate_quality(self, synthetic_data):
        """
        Evaluate the quality of synthetic data.
        
        Parameters
        ----------
        synthetic_data : pd.DataFrame
            The synthetic data to evaluate
        """
        from sdv.evaluation.single_table import evaluate_quality
        
        quality_report = evaluate_quality(
            real_data=self.real_data,
            synthetic_data=synthetic_data,
            metadata=self.metadata_schema
        )
        
        print("\n" + "="*50)
        print("QUALITY EVALUATION REPORT")
        print("="*50)
        print(f"Overall Quality Score: {quality_report.get_score():.3f}")
        print("\nDetailed Scores:")
        print(quality_report.get_details())
        
        return quality_report


# Optional: Load model later
# synthesizer_new = PADSDataSynthesizer(preprocessed_dir)
# synthesizer_new.load_model("pads_tvae_model.pkl")
# more_synthetic = synthesizer_new.generate_synthetic_data(num_samples=50)