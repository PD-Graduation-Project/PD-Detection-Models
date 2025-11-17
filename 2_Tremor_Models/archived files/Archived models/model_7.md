### 1. **Frequency Domain Analysis (FFT)** - HUGE for tremor!
- Tremor is **periodic** (4-6Hz for PD)
- Time-domain CNNs might miss this
- FFT directly captures frequency content
- Extracts power spectrum in 0.5-10Hz range

### 2. **Separate Super-Pathway for Dominant Hand**
- **Non-dominant**: 128-dim pathway (6→48→96→128)
- **Dominant**: 256-dim pathway (6→96→192→256) with squeeze-excitation
- **3x more parameters** for the hand that matters most!

### 3. **Statistical Moment Features**
- **Mean, Std**: Basic statistics
- **Skewness**: Tremor asymmetry
- **Kurtosis**: Spike detection (tremor has sharp peaks)
- PD tremor is less regular than normal movement

### 4. **Dilated Convolutions**
- `dilation=2` captures longer-range patterns
- Helps with rhythmic tremor detection

### 5. **Built-in Mixup Augmentation**
```python
# During training:
output = model(x, handedness, mixup_lambda=0.2)
```
This synthetically creates more training samples!

### 6. **Uncertainty Estimation**
```python
# Check model confidence
mean_logit, uncertainty = model.forward_with_uncertainty(x, handedness, num_samples=10)
# High uncertainty = model is confused = review this sample
```

### 7. **Bottleneck Architecture**
- Forces 512 → 128 → 256 compression
- Model must learn efficient representations
- Better generalization

### Training Tips:
```python
model = TremorNetAdvanced_V9(dropout=0.45, use_fft=True)

# Use mixup during training
for x, handedness, y in train_loader:
    if random.random() < 0.5:
        lambda_mix = np.random.beta(0.2, 0.2)  # Mixup strength
        output = model(x, handedness, mixup_lambda=lambda_mix)
    else:
        output = model(x, handedness)
```

This architecture is **specifically designed** for periodic, asymmetric, bilateral tremor data. The FFT + statistical features should capture what pure CNNs miss!