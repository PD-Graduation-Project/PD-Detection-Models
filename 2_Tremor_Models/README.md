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

- Input: 6 IMU channels -> 64 -> 128 -> 256 feature maps
- **Three** 1D convolutions with `stride = 2` (temporal downsampling by ~8×)
- Batch normalization and ReLU activations after each convolution
- Dropout applied for regularization

**Output shape:**
`[B, 256, T/8]` -> reshaped to `[B, T/8, 256]` for GRU input

---

## 2. GRU Block

- `input_size = 256` (from CNN output)
- 2 stacked **bidirectional** GRU layers (`hidden_size = 128`)
- Dropout applied between GRU layers

**Output shape:**
`[B, T/8, 2 * hidden_size]` = `[B, T/8, 256]` (bidirectional output)

---

## 3. Attention Block

The attention mechanism learns to **focus on the most diagnostically relevant time windows** rather than treating all time steps equally.

**Mechanism:**
  - Small attention network: `Linear(256→64) → Tanh → Linear(64→1)`
  - Produces scalar attention **scores** per **time step**
  - `Softmax` normalizes these scores across time
  - Weighted sum yields an attention-focused temporal representation

**Key advantage:** Automatically identifies *when* symptoms occur, ignoring irrelevant movement periods.

**Output shape:**
`[B, 256]` (attention-weighted temporal features vecor)

---

## 4. Global Pooling

In addition to attention, two pooling operations capture overall patterns:

- **Mean pooling:** `[B, 256]` (average across all time steps)
- **Max pooling:** `[B, 256]` (maximum activation across time)

These complement attention by providing different temporal summaries.

---

## 5. Wrist Embedding

- `nn.Embedding(2, wrist_embed_dim)` maps wrist type to a learned feature vector
- Default: 16-dimensional wrist embedding
- Integrates wrist context (left vs right) into the final classification

- **Think of it like this:**
  - For **left** wrist data, the model uses one embedding vector.
  - For **right** wrist data, it uses another.
  - During **training**, it learns how the tremor signals from the two wrists differ statistically or dynamically.

So effectively, those 32 parameters help the model link “wrist side” to movement patterns, improving classification consistency when left and right hands behave differently (which is common in PD).

---

## 6. Classifier

- Input: concatenation of attention features + mean pooling + max pooling + wrist embedding
- Input dimension:
    ```
    3 × (2 × hidden_size) + wrist_embed_dim
    = hidden_size * 6 + wrist_embed_dim
    = 128 * 6 + 16 = 784
    ```
- Hidden layer (128 units) with ReLU and dropout
- Final linear layer produces `num_classes` logits

---

## 7. Forward Pass

**Data Flow:**

```python
# 1. Input Formatting
x: [B, T, 6] -> [B, 6, T]                     # CNN expects channels-first format

# 2. CNN Feature Extraction
cnn_out: [B, 256, T/8]                        # local motion features after temporal downsampling

# 3. GRU Temporal Modeling
gru_in: [B, T/8, 256]
gru_out: [B, T/8, hidden_size*2]              # bidirectional GRU captures forward & backward dependencies

# 4. Temporal Feature Summarization
# The GRU outputs a sequence over time. Here we summarize it into fixed-size features:
# - Attention: focuses on the most informative time steps
# - Mean pooling: captures the average motion pattern
# - Max pooling: captures the strongest motion response
attended:    [B, hidden_size*2]               # attention-weighted representation
pooled_mean: [B, hidden_size*2]               # global average pooling
pooled_max:  [B, hidden_size*2]               # global max pooling
time_features = concat([attended, pooled_mean, pooled_max]) -> [B, hidden_size*6]

# 5. Wrist Context Embedding
wrist_embed: [B, 16]                          # learned wrist feature (left/right)

# 6. Fusion and Classification
combined = concat([time_features, wrist_embed]) -> [B, hidden_size*6 + wrist_embed_dim]
output = classifier(combined) -> [B, num_classes]            # final class logits
```

**Output:**
`[B, num_classes]`: suitable for `CrossEntropyLoss`

---

## 8. Full model architecture 
```
========================================================================================================================
Layer (type (var_name))                  Input Shape          Output Shape         Param #              Trainable
========================================================================================================================
TremorNetGRU (TremorNetGRU)              [1, 1024, 6]         [1, 3]               --                   True
├─Sequential (cnn)                       [1, 6, 1024]         [1, 256, 128]        --                   True
│    └─Conv1d (0)                        [1, 6, 1024]         [1, 64, 512]         1,984                True
│    └─BatchNorm1d (1)                   [1, 64, 512]         [1, 64, 512]         128                  True
│    └─ReLU (2)                          [1, 64, 512]         [1, 64, 512]         --                   --
│    └─Conv1d (3)                        [1, 64, 512]         [1, 128, 256]        24,704               True
│    └─BatchNorm1d (4)                   [1, 128, 256]        [1, 128, 256]        256                  True
│    └─ReLU (5)                          [1, 128, 256]        [1, 128, 256]        --                   --
│    └─Conv1d (6)                        [1, 128, 256]        [1, 256, 128]        98,560               True
│    └─BatchNorm1d (7)                   [1, 256, 128]        [1, 256, 128]        512                  True
│    └─ReLU (8)                          [1, 256, 128]        [1, 256, 128]        --                   --
│    └─Dropout (9)                       [1, 256, 128]        [1, 256, 128]        --                   --
├─GRU (gru)                              [1, 128, 256]        [1, 128, 256]        592,896              True
├─Sequential (attention)                 [1, 128, 256]        [1, 128, 1]          --                   True
│    └─Linear (0)                        [1, 128, 256]        [1, 128, 64]         16,448               True
│    └─Tanh (1)                          [1, 128, 64]         [1, 128, 64]         --                   --
│    └─Linear (2)                        [1, 128, 64]         [1, 128, 1]          65                   True
├─Embedding (wrist_embed)                [1]                  [1, 16]              32                   True
├─Sequential (classifier)                [1, 784]             [1, 3]               --                   True
│    └─Linear (0)                        [1, 784]             [1, 128]             100,480              True
│    └─ReLU (1)                          [1, 128]             [1, 128]             --                   --
│    └─Dropout (2)                       [1, 128]             [1, 128]             --                   --
│    └─Linear (3)                        [1, 128]             [1, 3]               387                  True
========================================================================================================================
Total params: 836,452
Trainable params: 836,452
Non-trainable params: 0
Total mult-adds (M): 95.96
========================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.90
Params size (MB): 3.35
Estimated Total Size (MB): 5.27
========================================================================================================================
```

