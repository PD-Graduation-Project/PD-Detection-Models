## Model Selection for Tabular Dataset (100,000 Samples)

Because we generated a large amount of tabular data (100,000 samples), we were able to safely use **larger models** to capture more complex feature interactions without overfitting. The selected models and their approximate number of parameters are:

- **DenseNet169** – ~14M parameters  
- **EfficientNet-B1** – ~7.8M parameters  
- **MobileNetV3-Large** – ~5.4M parameters  
- **ResNet18** – ~11.7M parameters

These models were chosen to leverage the increased dataset size and allow deeper or wider architectures to improve learning capacity while still maintaining generalization.
