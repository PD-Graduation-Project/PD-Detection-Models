from torch.utils.data import DataLoader, random_split
from .dataset import SyntheticTremorDataset
from pathlib import Path

def create_synthetic_dataloaders(pt_path, batch_size=32, train_split=0.8, print_details=True):
    dataset = SyntheticTremorDataset(pt_path)

    N = len(dataset)
    n_train = int(N * train_split)
    n_val = N - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    if print_details:
        print(f"Synthetic dataset loaded from {pt_path}")
        print(f"Total samples: {N}")
        print(f"Train: {n_train},  Val: {n_val}")
        print(f"Batch size: {batch_size}")

    return train_loader, val_loader


def create_synthetic_per_movement_dataloaders(root_dir, batch_size=32):
    root = Path(root_dir)
    movement_dataloaders = {}

    for pt_file in root.glob("*_synthetic.pt"):
        movement_name = pt_file.stem.replace("_synthetic", "")

        train_loader, val_loader = create_synthetic_dataloaders(
            pt_path=str(pt_file),
            batch_size=batch_size,
            print_details=False
        )

        movement_dataloaders[movement_name] = {
            "train": train_loader,
            "val": val_loader
        }

    return movement_dataloaders



