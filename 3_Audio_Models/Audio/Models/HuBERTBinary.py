import torch
import torch.nn as nn
import torchaudio

class HuBERTBinary(nn.Module):
    """
    HuBERT for binary audio classification.
    Similar architecture to Wav2Vec2 but different pretraining.
    
    Args:
        dropout_rate (float): Dropout probability
        hidden_units (list[int]): Hidden layer sizes
        freeze_feature_extractor (bool): Freeze feature extractor
    """
    def __init__(self, dropout_rate=0.3, hidden_units=[256, 128], freeze_feature_extractor=True):
        super().__init__()
        
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.hubert = bundle.get_model()
        self.sample_rate = bundle.sample_rate
        
        if freeze_feature_extractor:
            for param in self.hubert.feature_extractor.parameters():
                param.requires_grad = False
        
        in_features = 768
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
        features, _ = self.hubert.extract_features(x)
        pooled = features[-1].mean(dim=1)
        return self.classifier(pooled)
