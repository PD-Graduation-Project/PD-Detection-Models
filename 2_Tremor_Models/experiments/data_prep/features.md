### Key Features:
Per Second Windows (10 rows per recording)

- Each 1-second window (~102 samples) generates one row
- Creates 10 rows per subject-movement pair

### Feature Categories Extracted:

#### Statistical Features (per channel & magnitude):

- Basic: mean, std, var, min, max, range, median, MAD
- Moments: skewness, kurtosis
- Energy: RMS, total energy
- Percentiles: Q25, Q75, IQR
- Zero-crossing rate


#### Frequency Features (tremor-relevant):

- Frequency bands: delta (0.5-3Hz), tremor_low (3-6Hz), tremor_mid (6-9Hz), tremor_high (9-12Hz), beta, gamma
- Dominant frequency & power
- Spectral: centroid, spread, entropy, rolloff
- Power ratios per band


#### Temporal Features:

- Entropy, autocorrelation
- Velocity & acceleration statistics
- Peak detection & peak rate


#### Bimanual Features (left-right comparison):

- Correlation between hands
- Difference statistics
- Energy ratios
- Asymmetry index



### Output Structure:

- ~1500+ features per row
- Features for: 6 channels (accel x,y,z + gyro x,y,z) × 2 hands + magnitudes + bimanual
- Metadata: subject_id, label, handedness, movement type, clinical info