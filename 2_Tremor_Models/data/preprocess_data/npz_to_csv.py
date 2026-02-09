import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def npz_to_csv(
    data_path: str = "../../../project_datasets/tremor/movements",
    output_csv: str = "tremor_features.csv"
):
    """
    Convert all .npz files to a single CSV.

    Movements are encoded as integers [0–10].
    Features are named:
      - lh_1 ... lh_20
      - rh_1 ... rh_20
      - asym_1 ... asym_20
    """

    data_path = Path(data_path)
    all_rows = []

    # Map movement names to integers 0–10
    movement_names = sorted(
        [d.name for d in data_path.iterdir() if d.is_dir()]
    )
    movement_map = {name: idx for idx, name in enumerate(movement_names)}

    print("Loading .npz files...")

    for movement_dir in data_path.iterdir():
        if not movement_dir.is_dir():
            continue

        movement_name = movement_dir.name
        movement_id = movement_map[movement_name]

        for label_dir in movement_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_name = label_dir.name
            label = 0 if label_name == "Healthy" else 1

            for npz_file in tqdm(
                list(label_dir.glob("*.npz")),
                desc=f"{movement_name}/{label_name}"
            ):
                data = np.load(npz_file, allow_pickle=True)

                features = data["features"]
                handedness = int(data["handedness"])
                segment_idx = int(data.get("segment_idx", 0))

                if len(features) != 66:
                    raise ValueError(
                        f"{npz_file.name}: Expected 66 features, got {len(features)}"
                    )

                row = {
                    "movement": movement_id,
                    # "movement_name": movement_name,
                    "handedness": handedness,
                    "label": label,
                    # "label_name": label_name,
                    "segment_idx": segment_idx,
                }

                # Left hand features (1–22)
                for i in range(22):
                    row[f"lh_{i+1}"] = features[i]

                # Right hand features (1–22)
                for i in range(22):
                    row[f"rh_{i+1}"] = features[22 + i]

                # Asymmetry features (1–22)
                for i in range(22):
                    row[f"asym_{i+1}"] = features[44 + i]

                all_rows.append(row)

    print(f"\nCreating CSV with {len(all_rows)} rows...")
    df = pd.DataFrame(all_rows)

    # Column order
    metadata_cols = [
        "movement",
        # "movement_name",
        "handedness",
        "label",
        # "label_name",
        "segment_idx",
    ]

    feature_cols = (
        [f"lh_{i}" for i in range(1, 23)] +
        [f"rh_{i}" for i in range(1, 23)] +
        [f"asym_{i}" for i in range(1, 23)]
    )

    df = df[metadata_cols + feature_cols]

    df.to_csv(output_csv, index=False)

    print("=" * 60)
    print(f"CSV saved: {output_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())
    print("\nMovement mapping:")
    for k, v in movement_map.items():
        print(f"{v}: {k}")
    print("=" * 60)

    return df
