# Tremor Dataset Overview

- **Patients:** 469 total

  - 79 Healthy
  - 276 Parkinson’s Disease (PD)
  - 114 Other disorders

- **Movements:** 11 different types (each performed with both left and right wrists -> 22 total recordings)

- **Signal length:** 1024 or 2048 timepoints per file (movement-dependent)

- **Signal structure:** 6 IMU channels per timepoint (e.g., 3-axis accelerometer + 3-axis gyroscope)

---

## Dataset Preparation

- Each `.npz` file contains:

  - `signal`: array of shape `(N, 6)` -> IMU sensor readings
  - `label`: `0 = Healthy`, `1 = Parkinson's`, `2 = Other`
  - `wrist`: `0 = Left`, `1 = Right`
  - `subject_id`: unique identifier

### **Directory structure:**

  ```
  data_path/
  ├── Healthy/
  ├── Parkinson/
  └── Other/
  ```

### **`TremorDataset` class:**

  - Loads signals, wrist indicators, and labels from all groups
  - Keeps wrist as a **separate scalar tensor** (`0 = Left`, `1 = Right`)
  - Returns tuples of `(signal_tensor, wrist_tensor, label_tensor)`
  - Added `subject_ids` filter.
  ``` python
  # Example:
  # Say you have subjects: 1, 2, 3, 4, 5 in your data
  # And you split them: train=[1, 2, 3], val=[4, 5]

  # When you do:
  train_dataset = TremorDataset(
      data_path=data_path,
      subject_ids=[1, 2, 3],  # <-- can be list, set, or any iterable
      ...
  )
  # This dataset will ONLY load samples from subjects 1, 2, and 3
  # It will skip/ignore all .npz files from subjects 4 and 5

  val_dataset = TremorDataset(
      data_path=data_path,
      subject_ids=[4, 5],  # <-- different subjects
      ...
  )
  # This dataset will ONLY load samples from subjects 4 and 5
  # It will skip/ignore all .npz files from subjects 1, 2, and 3
  ```

### **`create_tremor_dataloaders()` function:**

  - Splits data into training and validation sets (stratified by label)
  - Returns ready-to-train PyTorch `DataLoader` objects

### Preprocessing **Steps**

To ensure clean and standardized input for the model, several preprocessing steps were applied:
  - **Resampling**: All signals were resampled or truncated to a fixed length of `1024` timepoints.
  - **Normalization**: Each IMU channel was normalized using **z-score normalization** (zero mean, unit variance) to remove amplitude bias across patients.
  - **Outlier removal**: Extremely noisy segments or files with missing readings were excluded.
  - **Label encoding**: Labels were remapped to binary form for the classification task (0 = Healthy, 1 = PD).
  - **Wrist encoding**: Wrist side was encoded as a separate scalar input to the model (0 = Left, 1 = Right).

---

# TODO:
- MAKE ACCURACY BETTER
- current problem: model collapsing (predecting all PD, then all healthy)