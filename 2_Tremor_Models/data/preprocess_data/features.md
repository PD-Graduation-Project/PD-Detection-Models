**catch22 features** - 22 canonical time-series features:

**Categories:**

1. **Distribution** (3 features)
   - Mean, standard deviation, skewness

2. **Autocorrelation** (5 features)
   - How signal correlates with itself over time
   - First minimum, decay rate, etc.

3. **Entropy/Complexity** (4 features)
   - Signal randomness/predictability
   - Permutation entropy, distribution entropy

4. **Outliers** (2 features)
   - Presence of extreme values

5. **Periodicity** (3 features)
   - Dominant frequency, spectral characteristics

6. **Nonlinearity** (3 features)
   - Chaos-like behavior, Lyapunov exponent

7. **Other** (2 features)
   - Transition statistics, fluctuation analysis

**Why these 22?**
- Selected from 7000+ possible features
- Capture different signal properties
- Minimal redundancy
- Computationally efficient

**For tremor:**
- Autocorrelation → rhythmicity
- Entropy → tremor regularity
- Distribution → amplitude variation

Full list: `pycatch22.catch22_all(signal)['names']`