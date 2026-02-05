import torch
import torch.nn as nn
import torchaudio

class Wav2Vec2Binary(nn.Module):
    """
    Wav2Vec2 for binary audio classification (Healthy vs PD).
    Uses pretrained wav2vec2 base model with frozen feature extractor.
    
    Args:
        dropout_rate (float): Dropout probability
        hidden_units (list[int]): Hidden layer sizes for classifier
        freeze_feature_extractor (bool): Freeze feature extractor weights
    """
    def __init__(self, dropout_rate=0.3, hidden_units=[256, 128], freeze_feature_extractor=True):
        super().__init__()
        
        # Load pretrained Wav2Vec2
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = bundle.get_model()
        self.sample_rate = bundle.sample_rate
        
        # Freeze feature extractor
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Classifier head
        in_features = 768  # wav2vec2 base hidden size
        classifier_layers = []
        
        for hidden_size in hidden_units:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, 1))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) waveform
        Returns:
            (batch, 1) logits
        """
        # Extract features
        features, _ = self.wav2vec2.extract_features(x)
        
        # Use last layer output, mean pool over time
        pooled = features[-1].mean(dim=1)  # (batch, 768)
        
        return self.classifier(pooled)
