# PD-Detection-Models
Multi-modal Parkinson’s Disease detection pipelines featuring models for spiral drawing, tremor, and audio-based analysis.

## Part 1: Spiral and wave drawings Model [DONE]
### Models compared:
![](1_Spiral_Drawing_Models/results/training/acc.png)

### Model used: 
`MobileNetV3LargeBinary` with modified classifier.

### Dataset size: 
- **2611**: Training
- **653**: Validation

### Number of trained epochs: 
- **20**: initial model comparisons
- **50**: fine-tuning with early stopping; planned 50.

### Metrics:
- *Accuracy*: **98.81%**
- *Recall*: **99.27%**
- *precision*: **98.30%**
- *F1-Score*: **98.75%**

![](1_Spiral_Drawing_Models/results/finetuning/output.png)

## Part 2: Tremor Model

---

## Part 3.1: Audio Model (Tabular)
### Models compared:
![](3_Audio_Models/Tabular/results/training/acc.png)

### Model used: 
`DenseNet169`.

#### Generated data (used in pre-training):
- Model used: TVAE (Tabular VAE)
- Generated data accuracy:
    - Column Shapes Score: **87.43%**
    - Column Pair Trends Score: **92.2%**
    - Overall Score (Average): **89.81%**
- Generated: 100k samples
    - **80K**: Training
    - **20K**: Validation

#### Real data (used in finetuning):
- **40**: Training
- **41**: Validation


### Metrics:
- Validation *Accuracy*: **100.00%**
- Validation *Recall*: **100.00%**
- Validation *precision*: **100.00%**
- Validation *F1-Score*: **100.00%**

![alt text](3_Audio_Models/Tabular/results/finetuning/output.png)

---
## Part 3.2: Audio Model (Spectogram) 

---

## Part 4: Subject's Metadata Model [DONE]
### Models compared:
![](4_User_Data_Model/results/training/acc.png)

### Model used: 
`DenseNet169`.

#### Generated data (used in pre-training):
- Model used: TVAE (Tabular VAE)
- Generated data accuracy:
    - Column Shapes Score: **91.69%**
    - Column Pair Trends Score: **87.36%**
    - Overall Score (Average): **89.53%**
- Generated: 100k samples
    - **80K**: Training
    - **20K**: Validation

#### Real data (used in finetuning):
- **284**: Training
- **71**: Validation


### Metrics:
- Validation *Accuracy*: **99.22%**
- Validation *Recall*: **99.06%**
- Validation *precision*: **100.00%**
- Validation *F1-Score*: **99.52%**

![alt text](4_User_Data_Model/results/finetuning/output.png)