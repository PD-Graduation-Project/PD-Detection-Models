## Tremor Dataset Overview

- **Patients:** 469 total
  - 79 Healthy
  - 276 Parkinson's Disease (PD) 
  - 114 Other disorders

- **Movements:** 11 different types (each performed with both left and right wrists, so 22 total)

- **Signal length:** 1024 or 2048 timepoints per file (movement-dependent)

- **Signal structure:** 6 IMU channels per timepoint

---

## Dataset Preparation

- Each `.npz` file contains:
  - `signal`: array of shape (N, 6) - IMU sensor values
  - `label`: 0=Healthy, 1=Parkinson's, 2=Other
  - `wrist`: 0=Left, 1=Right
  - `subject_id`: unique identifier

- Directory structure:
  - `Healthy/`
  - `Parkinson/`
  - `Other/`

- **`TremorDataset` class:**
  - Loads signals and labels from all groups
  - Adds wrist information as a **7th channel** to signals
  - Shuffles data for reproducibility
  - Returns `(signal_tensor, label_tensor)` pairs

- **`create_tremor_dataloaders()` function:**
  - Splits data into training/validation sets
  - Creates PyTorch DataLoader objects for batching

---

## Modeling Approach

- **22 individual recordings** (11 movements × 2 wrists) are grouped into **11 combined movements**
- Each movement type gets its **own specialized model**
- Models are trained to classify **Healthy vs PD vs Other** using their specific movement data
- An **ensemble method** combines predictions from all 11 models for final diagnosis