import torch
import torch.nn as nn
import torchaudio

class Wav2Vec2LargeBinary(nn.Module):
    """
    Wav2Vec2 LARGE for binary classification.
    More parameters, potentially better performance.
    
    Args:
        dropout_rate (float): Dropout probability
        hidden_units (list[int]): Hidden layer sizes
        freeze_feature_extractor (bool): Freeze feature extractor
    """
    def __init__(self, dropout_rate=0.3, hidden_units=[512, 256], freeze_feature_extractor=True):
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        self.wav2vec2 = bundle.get_model()
        self.sample_rate = bundle.sample_rate
        
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        in_features = 1024  # large model hidden size
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
        features, _ = self.wav2vec2.extract_features(x)
        pooled = features[-1].mean(dim=1)
        return self.classifier(pooled)