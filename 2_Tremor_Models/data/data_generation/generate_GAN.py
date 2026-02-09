import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality

def train_GaussianCopula(
    csv_dir: str,
    output_dir: str,
    create_metadata: bool = True
):
    """
    Train Gaussian Copula models on Parkinson's dataset
    (separate models for Healthy and PD).

    Returns:
        tuple:
            - GaussianCopulaSynthesizer (healthy)
            - GaussianCopulaSynthesizer (pd)
            - Metadata
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load real data
    real_df = _get_real_data(csv_dir)

    # 2. Create or load metadata
    metadata_path = os.path.join(output_dir, "tremor_metadata.json")

    if create_metadata and not os.path.isfile(metadata_path):
        metadata = Metadata.detect_from_dataframe(real_df)
        metadata.save_to_json(metadata_path)
    else:
        metadata = Metadata.load_from_json(metadata_path)

    # 3. Split data by class
    healthy_df = real_df[real_df["label"] == 0]
    pd_df = real_df[real_df["label"] == 1]

    # 4. Initialize synthesizers
    synth_healthy = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False
    )

    synth_pd = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=False
    )

    # 5. Train models
    print("Training Healthy Gaussian Copula synthesizer...")
    synth_healthy.fit(healthy_df)

    print("Training PD Gaussian Copula synthesizer...")
    synth_pd.fit(pd_df)

    # 6. Save models
    model_healthy = os.path.join(output_dir, "GC_healthy.pkl")
    model_pd = os.path.join(output_dir, "GC_pd.pkl")

    synth_healthy.save(model_healthy)
    print("Healthy synthesizer saved at", model_healthy)

    synth_pd.save(model_pd)
    print("PD synthesizer saved at", model_pd)

    return synth_healthy, synth_pd, metadata


def generate_tremor_data_GC(
    csv_dir: str,
    output_dir: str,
    file_name: str = "tremor_generated_gc",
    load_existing_model: bool = True,
    num_generated_samples: int = 20000,
    filter_data: bool = False,
    threshold_std: int = 3
):
    """
    Generate synthetic tremor data using Gaussian Copula models.
    """

    # 1. Load or train models
    if load_existing_model:
        model_healthy = os.path.join(output_dir, "GC_healthy.pkl")
        model_pd = os.path.join(output_dir, "GC_pd.pkl")
        metadata_path = os.path.join(output_dir, "tremor_metadata.json")

        synth_healthy = GaussianCopulaSynthesizer.load(model_healthy)
        print("Loaded Healthy GC synthesizer")

        synth_pd = GaussianCopulaSynthesizer.load(model_pd)
        print("Loaded PD GC synthesizer")

        metadata = Metadata.load_from_json(metadata_path)
        print("Loaded metadata")
    else:
        synth_healthy, synth_pd, metadata = train_GaussianCopula(
            csv_dir, output_dir
        )

    # 2. Generate balanced synthetic data
    half = num_generated_samples // 2

    synthetic_healthy = synth_healthy.sample(num_rows=half)
    synthetic_pd = synth_pd.sample(num_rows=half)

    synthetic_df = (
        pd.concat([synthetic_healthy, synthetic_pd], axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # 3. Save generated data
    synth_csv_path = os.path.join(output_dir, file_name + ".csv")
    synthetic_df.to_csv(synth_csv_path, index=False)
    print("Synthetic data saved at", synth_csv_path)

    # 4. Evaluate quality
    real_df = _get_real_data(csv_dir)
    report = evaluate_quality(
        real_data=real_df,
        synthetic_data=synthetic_df,
        metadata=metadata
    )

    print("\nOverall Quality Score:", report.get_score())
    print("Detailed Properties:")
    print(report.get_properties())
    print("-" * 40)

    # 5. Optional filtering
    if filter_data:
        filtered_df = _filter_synthetic_data(
            real_df,
            synthetic_df,
            metadata,
            threshold_std
        )

        filtered_path = os.path.join(
            output_dir, file_name + "_filtered.csv"
        )
        filtered_df.to_csv(filtered_path, index=False)
        print("Filtered data saved at", filtered_path)

        report = evaluate_quality(
            real_data=real_df,
            synthetic_data=filtered_df,
            metadata=metadata
        )

        print("\nFiltered Quality Score:", report.get_score())
        print(report.get_properties())

        return filtered_df

    return synthetic_df

def _get_real_data(csv_dir: str):
    """
    Load and preprocess dataset.
    Ensures all columns are numeric.
    """
    df = pd.read_csv(csv_dir)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df
