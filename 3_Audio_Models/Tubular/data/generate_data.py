import os
import pandas as pd

from ucimlrepo import fetch_ucirepo 
from clean_data import clean_columns

from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality

def get_real_data():
    """
    Load and preprocess the Parkinson's dataset from the UCI Machine Learning Repository.

    Steps performed:
        1. Fetch dataset (ID = 174).
        2. Merge feature and target DataFrames.
        3. Clean column names using clean_columns().
        4. Convert all columns to numeric (non-numeric values → NaN).

    Returns:
        pd.DataFrame: Preprocessed dataset combining features and targets.
    """
    
    # 1. Fetch dataset
    parkinsons = fetch_ucirepo(id=174)

    # 2. make one DataFrame with features + target(s)
    df = pd.concat([parkinsons.data.features, parkinsons.data.targets], axis=1)

    # 3. Clean column names
    df.columns = clean_columns(df.columns)

    # 4. Ensure numerics
    df = df.apply(pd.to_numeric, errors="coerce")
    
    return df

def train_TVAE(output_dir: str,
            epochs: int = 2000,
            create_metdata: bool = True):
    """
    Train a TVAE (Tabular VAE) model on the Parkinson's dataset.

    Args:
        output_dir (str): Folder where the model and metadata will be saved.
        epochs (int): number of epochs the synthesizer will train on real data for.
        create_metadata (bool): If True, generate metadata JSON from the DataFrame.

    Returns:
        tuple:
            - TVAESynthesizer: Trained TVAE model.
            - Metadata: Metadata object describing the dataset structure.
    """
    
    # 1. Load the dataset
    # ---------------------
    df = get_real_data()
    
    # 2. Create or load Metadata
    # -----------------------------
    metadata_path = os.path.join(output_dir, "voice_metadata.json")
    
    if create_metadata:
        metadata = Metadata.detect_from_dataframe(real_df)
        metadata.save_to_json(metadata_path)
    else:
        metadata = Metadata.load_from_json(metadata_path)
        
    # 3. Init TVAE synthesizer
    # --------------------------
    synth = TVAESynthesizer(metadata, epochs=epochs)
    
    # 4. Train on real data
    # ----------------------
    synth.fit(df)
    
    # 5. Save model
    # --------------
    model_path  = os.path.join(output_dir, "TVAE_model.pkl")
    synth.save(model_path)
    
    return synth, metadata


def generate_voice_data(output_dir: str,
                        load_existing_model: bool = True,
                        num_generated_samples: int = 20000):
    """
    Generate synthetic Parkinson's data using TVAE, save it, and evaluate quality.

    Args:
        output_dir (str): Directory containing trained model and metadata.
        load_existing_model (bool): If True, load pretrained TVAE model; otherwise retrain.
        num_generated_samples (int): Number of synthetic samples to generate.

    Returns:
        pd.DataFrame: The generated synthetic dataset.
    """
    
    # 1. Load or tain model
    # -----------------------
    if load_pretrained_model:
        model_path  = os.path.join(output_dir, "TVAE_model.pkl")
        metadata_path  = os.path.join(output_dir, "voice_metadata.json")
        
        synth = TVAESynthesizer.load(model_path )
        metadata = Metadata.load_from_json(metadata_path )
    else:
        synth, metadata = train_TVAE(output_dir)
    
    # 2. Generate data
    # -----------------
    synthetic_df = synth.sample(num_rows=num_generated_samples)
    
    # 3. save it as csv
    # ------------------
    synth_csv_path  = os.path.join(output_dir, "parkinsons_generated.json")
    synthetic_df.to_csv(synth_csv_path , index=False)
    
    # 4. Evaluate generated data
    # ----------------------------
    # Real dataset
    real_df = get_real_data()

    # Generated dataset
    synth_df = pd.read_csv(synth_csv_path )
    
    # Run evaluation
    report = evaluate_quality(
        real_data=real_df,
        synthetic_data=synth_df,
        metadata=metadata
    )

    # Print overall score (0 to 1, higher = better)
    print("Overall Quality Score:", report.get_score())

    # Detailed breakdown
    print("Detailed Properties:")
    print(report.get_properties())
    
    return synthetic_df