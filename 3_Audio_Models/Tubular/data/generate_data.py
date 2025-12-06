import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_visualization.plots import plot_class_counts

from ucimlrepo import fetch_ucirepo 

from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality

def _clean_columns(columns):
    """
    Clean and standardize column names.

    Actions performed:
        1. Replace spaces and ':' characters with underscores.
        2. Ensure all column names are unique by appending incremental suffixes.
    """
    seen_counts = {}
    cleaned_list = []
    for col in columns:
        cleaned = col.replace(":", "_").replace(" ", "_")
        if cleaned in seen_counts:
            seen_counts[cleaned] += 1
            cleaned = f"{cleaned}_{seen_counts[cleaned]}"
        else:
            seen_counts[cleaned] = 0

        cleaned_list.append(cleaned)
    return cleaned_list



def _get_real_data():
    """
    Load and preprocess the Parkinson's dataset from the UCI Machine Learning Repository.

    Steps performed:
        1. Fetch dataset (ID = 174).
        2. Merge feature and target DataFrames.
        3. Clean column names using _clean_columns().
        4. Convert all columns to numeric (non-numeric values → NaN).

    Returns:
        pd.DataFrame: Preprocessed dataset combining features and targets.
    """
    
    # 1. Fetch dataset
    parkinsons = fetch_ucirepo(id=174)

    # 2. make one DataFrame with features + target(s)
    df = pd.concat([parkinsons.data.features, parkinsons.data.targets], axis=1)

    # 3. Clean column names
    df.columns = _clean_columns(df.columns)

    # 4. Ensure numerics
    df = df.apply(pd.to_numeric, errors="coerce")
    
    return df



def train_TVAE(output_dir: str,
            epochs: int = 2000,
            create_metadata: bool = True):
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
    real_df = _get_real_data()
    
    # Plot class counts
    plot_class_counts(real_df, target_col="status")
    
    # 2. Create or load Metadata
    # -----------------------------
    metadata_path = os.path.join(output_dir, "voice_metadata.json")
    
    if create_metadata and not os.path.isfile(metadata_path):
        metadata = Metadata.detect_from_dataframe(real_df)
        metadata.save_to_json(metadata_path)
    else:
        metadata = Metadata.load_from_json(metadata_path)
        
    # 3. split into healthy vs PD (for balancing data)
    # ---------------------------------------------------
    healthy_df = real_df[real_df["status"] == 0]
    pd_df = real_df[real_df["status"] == 1]

    # 4. Init TWO TVAE synthesizers
    # ------------------------------
    synth_healthy = TVAESynthesizer(metadata, 
                                    epochs=epochs, 
                                    cuda=True, verbose=True)
    synth_pd = TVAESynthesizer(metadata, 
                                epochs=epochs, 
                                cuda=True, verbose=True)
    
    # 5. Train both
    # --------------
    print ("Training Healthy data synthesizer...")
    synth_healthy.fit(healthy_df)
    print ("Training PD data synthesizer...")
    synth_pd.fit(pd_df)
    
    # 6. Save both models
    # ---------------------
    model_healthy = os.path.join(output_dir, "TVAE_healthy.pkl")
    model_pd = os.path.join(output_dir, "TVAE_pd.pkl")

    synth_healthy.save(model_healthy)
    print ("Healthy synthesizer saved at", model_healthy)
    synth_pd.save(model_pd)
    print ("PD synthesizer saved at", model_pd)
    
    return synth_healthy, synth_pd, metadata



def generate_voice_data(output_dir: str,
                        file_name: str = "parkinsons_generated",
                        load_existing_model: bool = True,
                        num_generated_samples: int = 20000,
                        filter_data: bool = False,
                        threshold_std: int = 3):
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
    if load_existing_model:
        model_healthy = os.path.join(output_dir, "TVAE_healthy.pkl")
        model_pd = os.path.join(output_dir, "TVAE_pd.pkl")
        metadata_path  = os.path.join(output_dir, "voice_metadata.json")
        
        synth_healthy = TVAESynthesizer.load(model_healthy)
        print ("loaded Halthy synthesizer")
        synth_pd = TVAESynthesizer.load(model_pd)
        print ("loaded PD synthesizer")
        metadata = Metadata.load_from_json(metadata_path )
        print ("loaded Metdata")
    else:
        synth_healthy, synth_pd, metadata = train_TVAE(output_dir)
    
    # 2. Generate data (BALANCED)
    # -----------------------------
    half = num_generated_samples // 2

    synthetic_healthy = synth_healthy.sample(num_rows=half)
    synthetic_pd = synth_pd.sample(num_rows=half)

    synthetic_df = (
        pd.concat([synthetic_healthy, synthetic_pd], axis=0)
        .sample(frac=1, random_state=42)   # shuffle rows
        .reset_index(drop=True)
    )
    
    # 3. save it as csv
    # ------------------
    synth_csv_path  = os.path.join(output_dir, file_name+".csv")
    synthetic_df.to_csv(synth_csv_path , index=False)
    print ("Data generated and saved at", synth_csv_path)
    
    # 4. Evaluate generated data
    # ----------------------------
    real_df = _get_real_data()
    synth_df = pd.read_csv(synth_csv_path)
    
    report = evaluate_quality(
        real_data=real_df,
        synthetic_data=synth_df,
        metadata=metadata
    )

    print("Overall Quality Score:", report.get_score())
    print("Detailed Properties:")
    print(report.get_properties())
    print('-'*35)
    
    # 5. Filter data
    # ---------------
    if filter_data:
        filtered_df = _filter_synthetic_data(real_df, 
                                            synth_df, 
                                            metadata,
                                            threshold_std)
        
        synth_csv_path  = os.path.join(output_dir, file_name+"_filtered.csv")
        synthetic_df.to_csv(synth_csv_path , index=False)
        print ("Filtered generated data and saved at", synth_csv_path)
        
        # 4. Re-evaluate generated data after filtering
        # -----------------------------------------------
        report = evaluate_quality(
            real_data=real_df,
            synthetic_data=filtered_df,
            metadata=metadata
        )

        print("Overall Quality Score:", report.get_score())
        print("Detailed Properties:")
        print(report.get_properties())
        
        return filtered_df
    
    return synth_df


def _filter_synthetic_data(real_df, synthetic_df, metadata, threshold_std=10):
    """Remove outliers that are too far from real data distribution."""
    mask = pd.Series([True] * len(synthetic_df))
    
    for col in tqdm(real_df.columns, desc="Filtering outliers"):
        if col == 'status':
            continue
        
        real_mean = real_df[col].mean()
        real_std = real_df[col].std()
        
        # Keep samples within threshold standard deviations
        col_mask = (
            (synthetic_df[col] >= real_mean - threshold_std * real_std) &
            (synthetic_df[col] <= real_mean + threshold_std * real_std)
        )
        mask = mask & col_mask
    
    filtered_df = synthetic_df[mask].reset_index(drop=True)
    print(f"Kept {len(filtered_df)}/{len(synthetic_df)} samples ({len(filtered_df)/len(synthetic_df)*100:.1f}%)")
    
    return filtered_df