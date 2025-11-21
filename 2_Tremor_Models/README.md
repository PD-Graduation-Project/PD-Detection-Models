# Tremor Dataset Overview

* **Patients:** 469 total

  * 79 Healthy
  * 276 Parkinson’s Disease (PD)
  * 114 Other disorders

* **Movements:** **11 merged movement types**
  Each movement has **two recordings per sample**:

  * Left wrist
  * Right wrist
    → Stored together in a single `.npz` entry.

* **Signal length:** All signals are standardized to **1024 timepoints** (upsampled or downsampled).

* **Signal structure:**
  Each `.npz` file contains **two aligned IMU matrices**:

  ```
  left  wrist → (1024, 6)
  right wrist → (1024, 6)
  ```

  6 channels = 3-axis accelerometer + 3-axis gyroscope.

---

# Dataset Preparation

Each `.npz` file contains:

### **1. `signal`**

Shape: `(2, 1024, 6)`
Order is **always**:

```
signal[0] → left wrist IMU  
signal[1] → right wrist IMU  
```

**NOTE:** The dataset creation script **always saves (left, right)** regardless of subject handedness.

---

### **2. `label`**

```
0 = Healthy
1 = Parkinson's Disease
2 = Other disorders
```

---

### **3. `wrist`**

This is **subject handedness**, not the recording wrist:

```
0 = Left-handed subject
1 = Right-handed subject
```

**NOTE:** This is NOT which wrist the recording came from.
Recordings are always saved in (left_signal, right_signal) order.

---

### **4. `subject_id`**

A unique integer ID (3-digit ID from PADS dataset).

---

### **5. `metadata` vector**

Extracted from the dataset CSV and encoded numerically:

| Field                               | Meaning                       | Notes                                                      |
| ----------------------------------- | ----------------------------- | ---------------------------------------------------------- |
| `age_at_diagnosis`                  | PD diagnosis age              | -1 if missing                                              |
| `age`                               | current age                   | -1 if missing                                              |
| `height`                            | cm                            | -1 if missing                                              |
| `weight`                            | kg                            | -1 if missing                                              |
| `gender`                            | 0=male, 1=female, -1=unknown  | categorical encoded                                        |
| `appearance_in_kinship`             | tremor in family              | 1=True, 0=False, -1=missing                                |
| `appearance_in_first_grade_kinship` | tremor in first-degree family | 1/0/-1                                                     |
| `effect_of_alcohol_on_tremor`       | effect category               | Encoded as: 0=Unknown, 1=No effect, 2=Reduced, 3=Increased |

**NOTE:**

* All missing values become **-1**.
* All categorical variables are **mapped to integers** before saving.

---

# Directory Structure

Dataset is saved as **11 movement directories**, each containing 3 subfolders:

```
data_path/
├── CrossArms/
│   ├── Healthy/
│   ├── Parkinson/
│   └── Other/
├── FingerNose/
├── Rest/
├── ... (total 11 movements)
```

Inside each label folder:

```
001.npz
002.npz
...
```

Every `.npz` file corresponds to **one subject and one movement**
with **both wrist signals paired**.

---

# TremorDataset Class

The dataset now loads:

```
(signal_tensor, handedness_tensor, movement_tensor, label_tensor, metadata_tensor)
```

### **Major Updates**

* **Paired signals:**
  Signals have shape `(2, T, 6)` → `(Left, Right)`.

* **Handedness:**
  Stored as a scalar (0 or 1) representing **subject handedness**.

* **Movement index included:**
  Allows joint multi-movement training or per-movement loaders.

* **Metadata included:**
  Returned as an 8-dim vector.

* **Subject filtering included:**
  Splitting is done by **subject**, not samples.

---

### Example: Subject-based filtering

```python
train_dataset = TremorDataset(
    data_path=data_path,
    subject_ids=[1, 2, 3],
)

val_dataset = TremorDataset(
    data_path=data_path,
    subject_ids=[4, 5],
)
```

**NOTE:**
This prevents leakage because:

* All movements from subject X go to the same split.

---

# `create_tremor_dataloaders()` Function

### Features:

* Splits **by subject**, stratified by label.
* Handles **class imbalance** using `WeightedRandomSampler`.
* Can return:

  1. **Unified train/val loaders**
  2. **Per-movement loaders** (11 sets)

### Returned sample format

Each batch returns:

```
signal:   (B, 2, 1024, 6)
handedness: (B,)
movement:    (B,)
label:       (B,)
metadata:    (B, 8)
```

---

# Preprocessing Steps

All preprocessing now happens in `dataset_creation.py`.

### **1. Remove NaNs**

Replaces missing values with 0.

### **2. Clip outliers**

Range ±50 (empirically chosen for IMU sensors).

### **3. Low-pass Butterworth filter**

Cutoff = 10 Hz
Ensures tremor-related frequencies are preserved.

### **4. Resample**

Every signal is resampled to **1024 samples**.

**NOTE:**
Original dataset had lengths 1024 and 2048 depending on movement.
This step standardizes everything.

### **5. Normalize**

Z-score normalization per channel.

**NOTE:**
Normalization happens **after resampling**.

---

# Additional Notes (New)

### **(1) Signals are stored as (left, right) even for right-handed subjects**

This avoids confusion and makes the model's input stable.

### **(2) Handedness is a separate scalar**

Do NOT confuse with wrist order.

### **(3) Missing metadata is encoded as -1**

This allows the model to learn missingness patterns.

### **(4) Movement directories are merged into 11 categories**

The creation script handles the merging automatically.

### **(5) Per-movement dataloaders are supported**

Useful for movement-specific modeling.

---