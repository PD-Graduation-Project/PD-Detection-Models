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
- Best Model: `DenseNet121`
  - val_acc=0.9688
  - val_recall=1.0000
  - val_precision=0.9615
  - val_f1=0.9800

![alt text](results/real_data/output.png)