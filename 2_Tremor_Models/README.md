# Tremor Dataset Overview

- **Patients:** 469 total

  - 79 Healthy
  - 276 Parkinson’s Disease (PD)
  - 114 Other disorders

- **Movements:** 11 different types (each performed with both left and right wrists -> 22 total recordings)

- **Signal length:** 1024 or 2048 timepoints per file (movement-dependent)

- **Signal structure:** 6 IMU channels per timepoint

---

## Dataset Preparation

- Each `.npz` file contains:

  - `signal`: array of shape `(N, 6)` — IMU sensor readings
  - `label`: `0 = Healthy`, `1 = Parkinson's`, `2 = Other`
  - `wrist`: `0 = Left`, `1 = Right`
  - `subject_id`: unique identifier

- **Directory structure:**

  ```
  data_path/
  ├── Healthy/
  ├── Parkinson/
  └── Other/
  ```

- **`TremorDataset` class:**

  - Loads signals, wrist indicators, and labels from all groups
  - Keeps wrist as a **separate scalar tensor** (`0 = Left`, `1 = Right`)
  - Shuffles data for reproducibility
  - Returns tuples of `(signal_tensor, wrist_tensor, label_tensor)`

- **`create_tremor_dataloaders()` function:**

  - Splits data into training and validation sets (stratified by label)
  - Returns ready-to-train PyTorch `DataLoader` objects

---

## Modeling Approach

- **22 wrist-specific recordings** (11 movements × 2 wrists) are merged into **11 combined movement models**
- Each movement type has its **own specialized classifier**
- Each model predicts **Healthy / PD / Other** for its specific movement
- Final diagnosis is produced via an **ensemble** across all movement models

---

# Model V0: TremorNetGRU

## 1. CNN Block

- Input: 6 IMU channels -> 64 -> 128 feature maps
- Temporal downsampling using stride and padding
- Batch normalization and ReLU activations after each convolution
- Dropout applied for regularization

**Output shape:**
`[B, 128, T/4]` -> fed into GRU

---

## 2. GRU Block

- `input_size = 128` (from CNN output)
- 2 stacked bidirectional GRU layers (`hidden_size = 128`)
- Dropout applied between GRU layers

**Output shape:**
`[B, T/4, 2 * hidden_size]` (bidirectional output)

Last timestep extracted with:

```python
last_out = gru_out[:, -1, :]
```

---

## 3. Wrist Embedding

- `nn.Embedding(2, wrist_embed_dim)` maps wrist type to a learned feature vector
- Default: 16-dimensional wrist embedding
- Integrates wrist context into the final classification

---

## 4. Classifier

- Input: concatenation of GRU output and wrist embedding
- Hidden layer with ReLU and dropout
- Final linear layer produces `num_classes` logits

---

## 5. Forward Pass

**Data flow:**

```python
x: [Batch, Time, Channels] -> [B, C, T]  # for Conv1d
features: [B, 128, T/4] -> [B, T/4, 128]  # for GRU
```

**Concatenation:**

```python
combined = cat([last_out, wrist_embed], dim=1)
```

**Output:**
`[B, num_classes]`: suitable for `CrossEntropyLoss`

---

