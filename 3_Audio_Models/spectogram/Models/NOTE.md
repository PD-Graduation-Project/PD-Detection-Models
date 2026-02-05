## Model Selection for Spectrogram Dataset (82 Samples)

Because the spectrogram dataset is very small (only 82 images), we prioritized **smaller models** or models with moderate capacity to reduce the risk of overfitting. The selected models and their approximate number of parameters are:

- **DenseNet121** – ~8M parameters  
- **EfficientNet-B0** – ~5.3M parameters  
- **InceptionV3** – ~24M parameters (used as a benchmark in prior literature despite being large)  
- **ResNet18** – ~11.7M parameters  

These models were chosen to balance **feature extraction capability** with **overfitting risk** given the limited dataset size.
