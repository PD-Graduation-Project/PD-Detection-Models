## **7.1. Dense Layers with Dropout**
```python
for hidden_size in hidden_units:  # [512, 128]
    nn.Linear(in_features, hidden_size)
    nn.ReLU()
    nn.Dropout(dropout_rate)
```
**What it does:** Adds 2 hidden layers (1920 → 512 → 128) before final prediction.

**Why it helps:**
- **More layers = more learning capacity** to understand PD vs Healthy patterns
- **ReLU** = lets the model learn complex non-linear patterns
- **Dropout (0.5)** = randomly turns off 50% of neurons during training to prevent memorizing (overfitting)

**Simple analogy:** Instead of jumping straight from raw features to "PD or Healthy", the model now has intermediate "thinking" layers to process the information better.

---

## **7.2. Final Single Output**
```python
nn.Linear(in_features, 1)  # 128 → 1
```
**What it does:** Produces one number (logit) that represents confidence of PD.

**Why it helps:**
- Works with our binary loss function
- Sigmoid converts it to probability (0:1)
- Single output = simpler, faster

---

## **Overall Impact**

**Before (our old model):**
```
DenseNet features (1920) → [DIRECT] → Prediction (1)
```
-> Too direct, can't learn PD-specific patterns well

**After (improved model):**
```
DenseNet features (1920) → 512 → Dropout → 128 → Dropout → Prediction (1)
```
-> Gradual refinement, learns PD patterns better, less overfitting
