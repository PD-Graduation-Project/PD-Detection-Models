# PD-Detection-Models
Multi-modal Parkinson’s Disease detection pipelines featuring models for spiral drawing, tremor, and audio-based analysis.

## Part 1: Spiral and wave drawings model [DONE]
- Model used: `DenseNet50` with modified classifier.
- Dataset size: **2611** Training, **653** Validation.
- Number of trained epochs: **15** (initial model comparisons) + **43** (fine-tuning with early stopping; planned 50).
- Validation Accuracy: **97.02%**
- Validation Recall: **0.9787**
- Validation precision: **0.9609**
- Validation F1-Score: **0.9689**
- ![](1_Spiral_Drawing_Models/results/confusion_matrices/phase_3.png)

