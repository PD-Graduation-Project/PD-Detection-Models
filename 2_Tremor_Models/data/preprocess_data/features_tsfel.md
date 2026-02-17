# TSFEL Features & Parkinson's Disease Detection
================================================

TSFEL extracts **165 features** across 4 domains:
Statistical, Temporal, Spectral, and Fractal

---

## DOMAIN 1: STATISTICAL (~20 features)
*Describes the amplitude distribution of the tremor signal*

| Feature | What it measures | PD Relevance |
|---|---|---|
| **Mean** | Average signal value | PD tremor has higher mean magnitude than healthy |
| **Variance** | Spread of signal values | PD: high variance due to consistent oscillations |
| **Standard Deviation** | RMS deviation from mean | Higher in PD tremor |
| **Kurtosis** | Peakedness of distribution | PD tremor has sharper peaks → higher kurtosis |
| **Skewness** | Asymmetry of distribution | Tremor creates non-symmetric amplitude distributions |
| **Mean absolute deviation** | Average deviation from mean | Captures tremor amplitude robustly |
| **Median absolute deviation** | Robust spread measure | Less sensitive to outliers than std |
| **Root mean square (RMS)** | Signal energy | PD tremor = consistently high RMS |
| **Interquartile range** | Middle 50% spread | Captures core tremor amplitude |
| **Histogram bins (x5)** | Signal value distribution shape | PD tremor creates bimodal distributions (oscillating up/down) |
| **Entropy (statistical)** | Randomness of values | PD tremor is REGULAR → low entropy vs healthy movement |
| **Min / Max** | Signal extremes | Tremor amplitude range |
| **Peak-to-peak** | Difference between max and min | Direct tremor amplitude measure |

### Why statistical features matter for PD:
- Healthy movement = irregular, varied amplitudes → HIGH entropy, moderate variance
- PD tremor = regular oscillations → LOW entropy, HIGH variance, HIGH RMS
- **Asymmetry bonus:** Comparing left vs right statistical features reveals PD's unilateral tremor

---

## DOMAIN 2: TEMPORAL (~20 features)
*Describes patterns over time - rhythm and periodicity*

| Feature | What it measures | PD Relevance |
|---|---|---|
| **Autocorrelation** | Signal correlation with time-shifted version of itself | PD tremor at 4-6 Hz → STRONG autocorrelation at ~200ms lag |
| **Zero crossing rate** | How often signal crosses zero | PD tremor crosses zero ~8-12 times per second (4-6 Hz) |
| **Mean crossing rate** | How often signal crosses its mean | Regular PD tremor = consistent crossing rate |
| **Slope** | Linear trend of signal | Captures signal drift/fatigue over time |
| **Absolute energy** | Sum of squared values | Direct measure of total tremor power |
| **Area under curve (AUC)** | Sum of absolute values | What the paper used to detect more-affected hand! |
| **Peak detection** | Number, height, distance of peaks | PD: many regular equally-spaced peaks |
| **Signal distance** | Total path length of signal | Tremor = lots of movement → high signal distance |
| **Positive/Negative turning points** | Direction changes count | Regular tremor = regular direction changes |
| **Neighbourhood peaks** | Local maxima count | PD: consistent local maxima at tremor frequency |
| **Mean difference** | Mean of consecutive differences | Captures how fast signal changes |
| **Median difference** | Median of consecutive differences | Robust version of above |

### Why temporal features matter for PD:
- PD resting tremor frequency: **4-6 Hz** → zero crossing rate ~8-12/s
- PD postural tremor frequency: **4-8 Hz**
- Healthy voluntary movement: **irregular, unpredictable** crossings
- Autocorrelation is ESPECIALLY powerful → PD tremor repeats itself every ~200ms

---

## DOMAIN 3: SPECTRAL (~100 features)
*Frequency-domain analysis - THE MOST IMPORTANT for tremor*

| Feature | What it measures | PD Relevance |
|---|---|---|
| **FFT mean coefficient** | Average frequency content | Shifts toward 4-6 Hz in PD |
| **Fundamental frequency** | Dominant oscillation frequency | PD rest tremor: 4-6 Hz (very specific!) |
| **Max power spectrum** | Peak frequency power | High concentrated power at tremor frequency |
| **Spectral centroid** | Center of mass of spectrum | Shifts to tremor frequency band in PD |
| **Spectral spread** | Width of frequency content | PD tremor is narrow-band → low spread |
| **Spectral skewness** | Asymmetry of spectrum | PD has concentrated spectral peak |
| **Spectral kurtosis** | Peakedness of spectrum | High in PD (sharp frequency peak) |
| **Spectral entropy** | Spread of energy across frequencies | PD: LOW entropy (energy concentrated at tremor freq) |
| **Spectral flux** | Change in spectrum over time | PD tremor is stable → low spectral flux |
| **Spectral roll-off** | Frequency below which X% of energy lies | Captures tremor band energy concentration |
| **Power bandwidth** | Width of dominant frequency band | PD tremor: narrow band |
| **Human range energy** | Energy in 0.6-2.5 Hz (human movement) | PD tremor exceeds normal movement energy |
| **LPCC (Linear Prediction Cepstral Coefficients)** | Spectral envelope shape | Captures tremor harmonic structure |
| **MFCC (x13)** | Frequency band energies (like audio) | Each coefficient captures different frequency band |
| **Wavelet energy (x levels)** | Energy at different time-frequency scales | Captures tremor at multiple resolutions |
| **Wavelet entropy** | Complexity across frequency scales | PD: low wavelet entropy (regular, simple tremor) |
| **Power spectral density** | Energy distribution across frequencies | Full frequency profile |

### Why spectral features are the MOST important for PD:
```
PD Resting Tremor:    4-6 Hz  ← very specific signature
PD Postural Tremor:   4-8 Hz  ← also specific
Essential Tremor:     6-12 Hz ← different band (paper's task)
Healthy movement:     0-2 Hz  ← low frequency, no tremor peaks
```
- Spectral features directly capture this frequency signature
- `fundamental_frequency` alone can be highly discriminative
- `spectral_entropy` captures regularity vs irregularity

---

## DOMAIN 4: FRACTAL (~10 features)
*Complexity and chaos in movement*

| Feature | What it measures | PD Relevance |
|---|---|---|
| **Higuchi fractal dimension** | Signal complexity/roughness | PD tremor: LOWER complexity than healthy movement |
| **Petrosian fractal dimension** | Self-similarity at different scales | Tremor has predictable self-similar structure |
| **Detrended fluctuation analysis (DFA)** | Long-range correlations | PD shows altered long-range motor correlations |
| **Hurst exponent** | Persistence/memory in signal | PD tremor is more persistent than healthy movement |

### Why fractal features matter for PD:
- Healthy movement is **complex and unpredictable** (high fractal dimension)
- PD tremor is **simple and repetitive** (low fractal dimension)
- DFA captures how the nervous system loses complexity in PD

---

## SUMMARY: Most Discriminative Features for Your Task

| Rank | Feature | Why |
|---|---|---|
| 1 | **Fundamental frequency** | Direct tremor frequency (4-6 Hz) |
| 2 | **Spectral entropy** | Tremor = low, healthy = high |
| 3 | **Autocorrelation** | Rhythmicity of tremor |
| 4 | **RMS / AUC** | Tremor amplitude |
| 5 | **Zero crossing rate** | Frequency proxy in time domain |
| 6 | **Higuchi fractal dim** | Movement complexity |
| 7 | **MFCC** | Full frequency band profile |
| 8 | **Asymmetry features** | PD unilateral signature |

---

## For YOUR Setup (Left + Right )

With 1 magnitude channel per hand:
- Left hand: **165 features**
- Right hand: **165 features**
- **Total: 330 features**

The asymmetry features are especially powerful because:
- **Healthy:** left ≈ right → asymmetry ≈ 0
- **PD:** one hand much worse → asymmetry >> 0
- This is a known clinical marker of PD!