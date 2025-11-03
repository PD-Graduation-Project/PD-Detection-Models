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
  - Shuffles data for reproducibility
  - Returns tuples of `(signal_tensor, wrist_tensor, label_tensor)`

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
- ADD HANDEDNESS!!!!!
- find a better way to handle the class imbalance
    - maybe by removing some PD data and try to make it closer to 50% (now it's 77% !!)
    - try GANs on healthy data?
- try models for each movement, and remove movements that are very hard to detect