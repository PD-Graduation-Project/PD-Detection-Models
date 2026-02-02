# Tremor Detection Pipeline - Simple Workflow

## Overview
This system detects and measures tremor severity from IMU sensor data (accelerometer + gyroscope) from both hands.

---

## Input
- **x**: `[B, 2, T, 6]` - IMU data from both hands
  - B = batch size
  - 2 = two hands (left & right)
  - T = 1024 samples (~10 seconds at 100Hz)
  - 6 = 3 accelerometer axes + 3 gyroscope axes
- **handedness**: `[B]` - dominant hand (0=left, 1=right)
- **movements**: `[B]` - movement type (optional)

## Output
- **Tremor severity**: `[B, 1]` - predicted tremor score

---

## Pipeline Workflow

### 1. **Hand Separation** → `[B, 6, T]` each
Splits input into left and right hand signals.
- **Why**: Each hand needs separate analysis before comparing them.

### 2. **Multi-Scale CNN** → `[B, 128]`
Extracts features using 3 parallel convolutions:
- Fast (kernel=3): Catches rapid tremors
- Medium (kernel=7): Catches moderate tremors  
- Slow (kernel=15): Catches slow tremors

**Process**:
- Concatenates all scales → `[B, 192, T/2]`
- Further convolution → `[B, 128, T/4]`
- Squeeze-Excitation (channel attention) → highlights important frequency patterns
- Temporal attention pooling → `[B, 128]`

**Why**: Tremors happen at different speeds. This captures all tremor frequencies simultaneously.

### 3. **Dominant Hand Weighting** → `[B, 128]`
Weights CNN features:
- Dominant hand: 3.0x weight
- Non-dominant hand: 1.0x weight
- Combined weighted average → `[B, 128]`

**Why**: Dominant hand tremors are more clinically significant for diagnosis.

### 4. **Frequency Analysis** → `[B, 11, 128]` → `[B, 128]`
For each second (10 segments) + full signal (1 segment):
- Computes spectrograms (time-frequency representation)
- CNN encodes patterns → `[B, output_dim/2]`
- Extracts 5 frequency band energies (0.5-3Hz, 3-6Hz, etc.)
- Computes left-right coherence per band
- Pools features → `[B, output_dim/2]`
- Concatenates → `[B, output_dim]`
- Attention pools over 11 segments → `[B, 128]`

**Why**: Different tremor types have characteristic frequencies (Parkinson's ~4-6Hz, Essential ~6-12Hz). Coherence shows if hands shake together (indicates certain conditions).

### 5. **Statistical Features** → `[B, 64]`
Computes 7 statistics for accelerometer and gyroscope:
- Mean, Std, Max, Min, RMS, Skewness, Kurtosis
- Accelerometer stats: `[B, 7]`
- Gyroscope stats: `[B, 7]`
- Projects to: `[B, 32]` each → concatenated to `[B, 64]`

**Why**: Captures overall signal properties (intensity, variability, peak amplitudes) that complement frequency analysis.

### 6. **Bilateral Coordination** → `[B, 64]`
- Gets intermediate CNN sequences: `[B, SeqLen, 128]` for each hand
- Concatenates sequences: `[B, 2×SeqLen, 128]`
- Self-attention: learns relationships between left/right movements over time
- Pools and projects → `[B, 64]`

**Why**: Shows how both hands coordinate over a period of time instead of through a compressed vector. Pathological tremors often have specific coordination patterns.

### 7. **Hand Asymmetry** → `[B, 64]`
- Computes absolute difference between left and right CNN features
- Projects → `[B, 64]`

**Why**: Asymmetric tremors (stronger in one hand) indicate specific neurological conditions.

### 8. **Hand Embedding** → `[B, 48]`
- Embeds handedness information
- Projects through MLP → `[B, 48]`

**Why**: Encodes which hand is dominant as learnable features.

### 9. **Movement Embedding** → `[B, 32]` (if enabled)
- Embeds movement type (e.g., rest, postural, kinetic)
- Projects and normalizes → `[B, 32]`

**Why**: Tremor characteristics vary by movement type. This context improves prediction.

### 10. **Feature Fusion** → `[B, 1]`
Concatenates all features:
- CNN: 128
- Frequency: 128
- Statistical: 64
- Bilateral: 64
- Asymmetry: 64
- Hand: 48
- Movement: 32 (optional)
- **Total**: 528 dimensions

Passes through deep MLP:
- `528 → 640 → 192 → 320 → 128 → 1`
- Uses BatchNorm, GELU, and Dropout for regularization

**Final output**: `[B, 1]` tremor severity score

---

## Key Design Choices

### Why Multiple Feature Types?
- **CNN**: Learns complex temporal patterns
- **Frequency**: Medical tremor signatures are frequency-specific
- **Statistical**: Captures amplitude and distribution characteristics
- **Bilateral**: Inter-hand coordination is clinically diagnostic
- **Asymmetry**: Lateralization indicates specific pathologies

### Why Per-Second Analysis in Frequency?
Tremors can vary within a 10-second window. Per-second analysis captures temporal dynamics that full-window analysis would miss.

### Why Dominant Hand Weighting?
Clinical diagnosis focuses more on dominant hand tremors as they're more functionally impactful and diagnostically relevant.

---

## Summary Flow

```
Input [B,2,T,6] 
  ↓
Split Hands → [B,6,T] × 2
  ↓
├─→ Multi-Scale CNN → [B,128]
├─→ Frequency Analysis → [B,128]  
├─→ Statistical Features → [B,64]
├─→ Bilateral Coordination → [B,64]
├─→ Asymmetry → [B,64]
├─→ Hand Embedding → [B,48]
└─→ Movement Embedding → [B,32]
  ↓
Concatenate → [B,528]
  ↓
Deep MLP Fusion
  ↓
Output [B,1] tremor severity
```