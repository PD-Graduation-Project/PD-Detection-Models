import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Parkinson Dataset
# -------------------
class ParkinsonDataset(Dataset):
    """
    A PyTorch Dataset for the Parkinson CSV dataset.

    Args:
        csv_path (str): Path to the CSV file.

    Behavior:
        - Loads the CSV using pandas.
        - One-hot encodes 'gender' column.
        - Features = all columns except 'label'.
        - Target = 'label'.
        - Converts both to float32 PyTorch tensors.
    """
    def __init__(self, csv_path: str):
        super().__init__()

        # 1. Load CSV file
        df = pd.read_csv(csv_path)

        # 2. Convert boolean columns to integers (True/False/-1 -> 1/0/-1)
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                # Check if column contains boolean-like strings
                unique_vals = df[col].astype(str).unique()
                if any(val in ['True', 'False', 'true', 'false', '-1'] for val in unique_vals):
                    df[col] = df[col].astype(str).map({
                        # Trues
                        'True': 1, 
                        'true': 1, 
                        # Falses
                        'False': 0, 
                        'false': 0, 
                        # Others
                        '-1': -1,
                        '-1.0': -1
                    })
                    print(f"✓ Converted boolean column '{col}' to numeric (True=1, False=0, -1=-1)")

        # 3. Encode gender explicitly (male=1, female=0)
        if 'gender' in df.columns:
            # Replace this section with explicit mapping
            df['gender'] = df['gender'].map({
                # Males
                'male': 1,      # or whatever represents male in your CSV
                'Male': 1,
                # Females
                'female': 0,    # or whatever represents female in your CSV
                'Female': 0,
                # Others
                -1: -1          # handle missing values
            })
            # Fill any unmapped values with -1
            df['gender'] = df['gender'].fillna(-1)
            
        print(f"✓ Converted 'Gender' column to numeric (Male=1, Female=0)")

        # 4. Separate features and labels
        self.y = df['label'].values.astype('float32')
        cols_to_drop = ['label', 'id']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        # Keep column names before converting to numpy
        feature_df = df.drop(columns=cols_to_drop)
        feature_columns = feature_df.columns.tolist()
        
        self.X = feature_df.values.astype('float32')

        # 5. **Handle NaN values BEFORE normalization**
        nan_mask = pd.isna(self.X)
        if nan_mask.any():
            nan_cols = nan_mask.any(axis=0)
            nan_column_names = [feature_columns[i] for i, has_nan in enumerate(nan_cols) if has_nan]
            nan_counts = {feature_columns[i]: nan_mask[:, i].sum() for i in range(len(feature_columns)) if nan_cols[i]}
            
            print(f"⚠️  Warning: Found {nan_mask.sum()} total NaN values across {len(nan_column_names)} columns:")
            for col_name, count in nan_counts.items():
                print(f"   - {col_name}: {count} NaN values")
            
            # Fill NaN with column mean
            col_means = pd.DataFrame(self.X, columns=feature_columns).mean(axis=0)
            self.X = pd.DataFrame(self.X, columns=feature_columns).fillna(col_means).values.astype('float32')
            print(f"✓ Filled NaN values with column means")

        # 6. Normalizing data
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X)

        # 7. Convert to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        
    # Required Dataset method
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Function: Create Train/Test DataLoaders
# -----------------------------------------
def create_dataloaders(
        csv_path: str,
        train_val_split: float = 0.8,
        batch_size: int = 16,
        random_seed: int = 42
    ):
    """
    Creates training and validation DataLoaders from the CSV dataset.

    Args:
        csv_path (str): Path to CSV file.
        train_val_split (float): Fraction for training (e.g., 0.8 → 80% train).
        batch_size (int): Batch size for DataLoaders.
        random_seed (int): Seed for reproducible splitting.

    Returns:
        (train_loader, val_loader)
    """
    # 1. Load full datase
    dataset = ParkinsonDataset(csv_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # 2. labels needed for stratified split
    labels = dataset.y.numpy().astype(int)
    
    # 3. Stratified train/validation spli
    train_indices, val_indices = train_test_split(
        indices,
        test_size=1 - train_val_split,
        stratify=labels,
        random_state=random_seed
    )
    
    # 4. Create Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # 5. Create DataLoader
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,         # shuffle training batches
        pin_memory=True,      # performance optimization
        drop_last=True        # ensures fixed batch size
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,        # no shuffling for validation/testing
        pin_memory=True
    )
    
    # 6. Debug informatio
    print("=========================================")
    print(f"Train dataset size:      {len(train_subset)}")
    print(f"Validation dataset size: {len(val_subset)}")
    print("-----------------------------------------")
    print(f"Train dataloader steps:  {len(train_dataloader)}")
    print(f"Val dataloader steps:    {len(val_dataloader)}")
    print("=========================================")

    return train_dataloader, val_dataloader
