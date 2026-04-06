# Feature Description (PD Tremor Detection)

This document describes the **27 features** extracted per signal segment.  
They include:
- 22 catch22 features (time-series dynamics)
- 2 basic statistical features (signal distribution)
- 3 STFT-based features (frequency-specific tremor)


---

# 1–22: catch22 Features

## Distribution & Amplitude

### 1. DN_HistogramMode_5
- **What**: Mode using 5-bin histogram  
- **How**: Most frequent amplitude bin  
- **Why (PD)**: Detects amplitude bias from tremor energy  

### 2. DN_HistogramMode_10
- Same as above with finer resolution  
- Captures subtle amplitude differences  

---

## Outlier Structure

### 3. DN_OutlierInclude_p_001_mdrmd
- **What**: Timing of positive outliers  
- **How**: Distribution of large spikes  
- **Why**: Tremor causes structured bursts  

### 4. DN_OutlierInclude_n_001_mdrmd
- Same for negative outliers  
- Captures asymmetric tremor patterns  

---

## Autocorrelation / Temporal Memory

### 5. CO_f1ecac
- **What**: Autocorrelation decay time  
- **How**: First 1/e crossing  
- **Why**: PD → slower dynamics  

### 6. CO_FirstMin_ac
- **What**: First minimum of autocorrelation  
- **Why**: Captures periodicity (tremor cycles)  

---

## Frequency Content

### 7. SP_Summaries_welch_rect_area_5_1
- **What**: Low-frequency power  
- **How**: Welch PSD integration  
- **Why**: Tremor lives in low-frequency band  

### 8. SP_Summaries_welch_rect_centroid
- **What**: Spectral centroid  
- **Why**: PD shifts energy toward tremor band  

---

## Predictability

### 9. FC_LocalSimple_mean3_stderr
- **What**: Forecast error  
- **How**: Rolling mean prediction  
- **Why**: PD signals become more predictable or rigid  

---

## Increment Dynamics

### 10. CO_trev_1_num
- **What**: Time irreversibility  
- **Why**: Healthy signals are more asymmetric  

### 11. CO_HistogramAMI_even_2_5
- **What**: Automutual information  
- **Why**: Measures nonlinear dependence  

---

## Fluctuations

### 12. MD_hrv_classic_pnn40
- **What**: Large step changes ratio  
- **Why**: PD reduces variability  

### 13. SB_BinaryStats_mean_longstretch1
- **What**: Longest above-mean segment  
- **Why**: Detects sustained tremor  

### 14. SB_BinaryStats_diff_longstretch0
- **What**: Longest decreasing trend  
- **Why**: Captures movement smoothness  

---

## Symbolic / Entropy

### 15. SB_MotifThree_quantile_hh
- **What**: Symbolic pattern entropy  
- **Why**: PD reduces complexity  

### 16. SB_TransitionMatrix_3ac_sumdiagcov
- **What**: Transition structure  
- **Why**: Captures repetitive motion  

---

## Nonlinear Dynamics

### 17. CO_Embed2_Dist_tau_d_expfit_meandiff
- **What**: Embedding space structure  
- **Why**: PD reduces dynamical richness  

### 18. IN_AutoMutualInfoStats_40_gaussian_fmmi
- **What**: AMI decay scale  
- **Why**: Measures dependency over time  

---

## Periodicity

### 19. PD_PeriodicityWang_th0_01
- **What**: Periodicity strength  
- **Why**: Strong indicator of tremor  

---

## Scaling / Fractality

### 20. SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
- **What**: Rescaled range analysis  
- **Why**: PD disrupts long-term structure  

### 21. SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
- **What**: Detrended fluctuation analysis  
- **Why**: Loss of fractal behavior in PD  

---

## Whitening / Correlation Change

### 22. CO_Embed2_Dist_tau_d_expfit_meandiff
- **What**: Change after differencing  
- **Why**: Captures micro-dynamics  

---

# 23–24: Basic Statistical Features (`catch24=True`)

### 23. Mean
- **What**: Average signal value  
- **Why**:
  - Detects bias / offset  

### 24. Standard Deviation
- **What**: Signal variability  
- **Why**:
  - PD often reduces variability (rigidity)  

---

# 25–27: STFT Features (Frequency-Specific)

### 25. Tremor Power Ratio
- **What**: Energy in 3–8 Hz band / total energy  
- **How**:
  - STFT → spectrogram  
  - Sum power in tremor band  
- **Why**:
  - Direct measure of tremor strength  

---

### 26. Tremor Stability
- **What**: Std of tremor-band power over time  
- **How**:
  - Compute band power per frame  
  - Take standard deviation  
- **Why**:
  - PD tremor can be intermittent  
  - Captures temporal consistency  

---

### 27. Peak Frequency
- **What**: Dominant frequency  
- **How**:
  - Average spectrum → argmax  
- **Why**:
  - PD tremor typically ~4–6 Hz  

---



# Summary

These features collectively capture:

| Property | Captured by |
|----------|------------|
| Amplitude | Histogram modes, mean, std |
| Frequency | STFT, Welch features |
| Periodicity | Autocorrelation, periodicity |
| Variability | pnn40, std |
| Complexity | Entropy, DFA, AMI |
| Dynamics | Embedding, irreversibility |

**Key insight:**  
PD tremor is characterized by:
- Strong **low-frequency periodicity**
- Reduced **complexity**
- Altered **temporal dynamics**

This feature set explicitly targets all three.