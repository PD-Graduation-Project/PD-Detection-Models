# Audio Classification - Model Selection

The audio dataset contains raw .wav files for Parkinson's Disease (PD) detection.  
We use **self-supervised pretrained audio models** from torchaudio that were trained on large speech datasets, then fine-tune them for binary classification (Healthy vs PD).

## Models Chosen

### **Wav2Vec2 Base** – ~95M parameters
- **Approach**: Self-supervised learning on unlabeled speech data
- **How it works**: 
  - Uses CNN feature extractor to convert raw waveform into latent representations
  - Transformer encoder processes these features with context
  - Pretrained via contrastive learning (mask prediction task)
  - We freeze the feature extractor and only train the classifier head
- **Best for**: General-purpose speech tasks, good baseline

### **HuBERT Base** – ~95M parameters  
- **Approach**: Hidden-Unit BERT for speech
- **How it works**:
  - Similar architecture to Wav2Vec2 but different pretraining
  - Uses k-means clustering to create pseudo-labels for masked prediction
  - Learns better discrete speech units through iterative refinement
  - More robust to acoustic variations than Wav2Vec2
- **Best for**: Tasks requiring fine-grained acoustic feature understanding

### **WavLM Base** – ~95M parameters
- **Approach**: Improved Wav2Vec2 specifically for speech tasks
- **How it works**:
  - Builds on Wav2Vec2 with gated relative position bias
  - Uses denoising pretraining (trained on noisy speech)
  - Better at handling speaker variation and background noise
  - Improved temporal modeling for speech-specific tasks
- **Best for**: Robust speech classification with real-world audio conditions

## Training Strategy

All models use:
- **Frozen feature extractors** to prevent overfitting on small datasets
- **Mean pooling** over time dimension to get fixed-size representations
- **Custom classifier heads** with dropout for binary classification
- **Sample rate**: 16kHz (standard for speech models)