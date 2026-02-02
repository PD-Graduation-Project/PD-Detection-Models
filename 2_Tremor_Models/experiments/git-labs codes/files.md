## Preprocessing
### `l1_trend_filter.py` - What It Does:
- This file removes noise from sensor signals while preserving important trends and changes.
- Think of it like drawing a smooth line through noisy data, but:
    - It keeps sharp changes (sudden movements, tremors)
    - It removes random jitter (sensor noise)
- Math simplified:
```
Find smoothed signal x that minimizes:
    (1) How different x is from original signal y
    (2) How "jumpy" x is (but allows some jumps via L1 penalty)

vlambda controls the tradeoff:
    - High vlambda → Very smooth (removes tremors too!)
    - Low vlambda → Keeps noise
    - Sweet spot (50) → Removes noise, keeps tremors
```
---

## **`run_preprocessing.py` - What It Does**

This script takes **raw sensor data** and prepares it for machine learning by:
1. Loading raw movement data
2. Organizing/sorting channels
3. Removing unwanted tasks
4. **Applying L1 trend filter** to remove gravity
5. Saving cleaned data

---

### STEPS

#### **Step 1: Load Data (Lines 15-16)**
```python
id = df['subject_id'][0]
data, channels = get_data_from_observation(movement_dir, df)
```
- Loads raw sensor data for one subject
- `channels` = list of column names like `"Relaxed1_LeftWrist_Accelerometer_X"`

---

#### **Step 2: Sort Channels (Lines 18-31)**
```python
for task in ['Relaxed1', 'Relaxed2', 'RelaxedTask1', ...]:
    for wrist in ['LeftWrist', 'RightWrist']:
        for sensor in ['Time', 'Accelerometer', 'Gyroscope']:
```

**What's happening**: The data has 14 different **tasks** (exercises):
- **Relaxed**: Arms resting
- **StretchHold**: Hold arms stretched out
- **HoldWeight**: Hold a weight
- **DrinkGlas**: Drinking motion
- **CrossArms**: Cross arms over chest
- **TouchNose**: Touch nose with finger
- **Entrainment**: Rhythmic tapping

Each task has:
- Left wrist + Right wrist sensors
- Accelerometer (X, Y, Z axes)
- Gyroscope (X, Y, Z axes)
- Time stamps

This creates a consistent order across all subjects.

---

#### **Step 3: Remove Unwanted Data (Lines 35-37)**
```python
to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
keep_mask = ~pd.Series(channels).str.contains(to_remove)
```

**Removes**:
- **Time** columns (not needed for ML)
- **LiftHold, PointFinger, TouchIndex** tasks (probably not useful for classification)

**Why remove tasks?** Some exercises might not show PD symptoms well, so they discard them.

---

#### **Step 4: Apply L1 Trend Filter (Lines 47-49)**
```python
to_process = 'Accelerometer'
# Remove gravitational offset
data[process_mask, :] = np.apply_along_axis(
    lambda x: x - l1_trend_filter(x, vlambda=50, verbose=False), 1,
    data[process_mask, :]
)
```

**Critical step!** For each accelerometer channel:
1. Apply L1 trend filter (gets the slow-moving gravity component)
2. **Subtract it** from the original signal
3. Result: Only movement acceleration (tremors, gestures), no gravity

**Why?** Accelerometers measure **gravity + movement**. Gravity is ~9.8 m/s² pointing down and masks the actual movement patterns.

---

#### **Step 5: Remove Vibration Noise (Line 51)**
```python
data = data[:, 48:]
```
- Removes first 0.5 seconds (48 samples at ~96 Hz)
- **Why?** The smartwatch vibrates to notify the start of a task, creating artifacts

---

#### **Step 6: Save to Binary File (Line 52)**
```python
data.tofile(f'{mov_path}{id}_ml.bin')
```
- Saves as `.bin` (binary format, smaller & faster than CSV)
- Filename: `001_ml.bin`, `002_ml.bin`, etc.

---

### **MAIN EXECUTION BLOCK (Lines 55-71)**

#### **Create Label File (Lines 56-65)**
```python
df['label'] = df['condition']
df.replace({'label': {'Healthy': 0,
                      "Parkinson's": 1,
                      'Other Movement Disorders': 2,
                      ...}})
df.to_csv(f'{data_path}file_list.csv')
```

**Creates a CSV** mapping subjects to labels:
- **0** = Healthy
- **1** = Parkinson's Disease
- **2** = Other disorders (Essential Tremor, MS, etc.)

This is your **ground truth** for training!

---

#### **Process All Subjects (Lines 67-71)**
```python
for df_element in df_list:
    preprocess_movement(df_element)
```
- Loops through all subjects
- Applies preprocessing to each one

---

## **`data_handling.py` - What It Does**
- This file contains helper functions to load different types of data from the PADS dataset. Think of it as the "data reader" that knows how to parse the specific file formats used in this project.

---

