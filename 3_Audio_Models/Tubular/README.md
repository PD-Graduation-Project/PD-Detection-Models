## Dataset used: 
[Parkinsons dataset — UCI ML Repository (ID 174)](https://archive.ics.uci.edu/dataset/174/parkinsons)

![alt text](data/data_visualization/image.png)

---

## GAN used:
- TVAE (Tabular VAE)

```
(1/2) Evaluating Column Shapes: |██████████| 23/23 [00:00<00:00, 139.02it/s]|
Column Shapes Score: 83.06%

(2/2) Evaluating Column Pair Trends: |██████████| 253/253 [00:02<00:00, 88.52it/s]|
Column Pair Trends Score: 92.87%

Overall Score (Average): 87.97%

Overall Quality Score: 0.8796556104364323
Detailed Properties:
             Property     Score
0       Column Shapes  0.830607
1  Column Pair Trends  0.928704
-----------------------------------
```

---

## Model Architectures:
- `ResNet18`: 11.7M
- `DenseNet121`: 8M
- `EfficientNet-B0`: 5M
- `MobileNet_V2`: 2.5M
- `MobileNet_V3`: 5.5M

---

## Results:
### 1. WITHOUT PRE-TRAINING (real data only)
- Best Model: `DenseNet121`
  - val_acc=0.9688
  - val_recall=1.0000
  - val_precision=0.9615
  - val_f1=0.9800

![alt text](results/real_data/output.png)

### 2. PRE-TRAINING (generated data plus real data) (80-20 split)
- Best Model: `EfficientNet-B0`
  - val_loss=0.0001
  - val_acc=1.0000
  - val_recall=1.0000
  - val_precision=1.0000
  - val_f1=1.0000

![alt text](results/real_and_generated/80-20_split/output.png)

### 3. PRE-TRAINING (generated data plus real data) (50-50 split)
- Best Model: `EfficientNet-B0`
  - val_loss=0.0032
  - val_acc=1.0000
  - val_recall=1.0000
  - val_precision=1.0000
  - val_f1=1.0000

![alt text](results/real_and_generated/50-50_split/output.png)