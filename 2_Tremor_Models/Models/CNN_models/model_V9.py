import torch
from torch import nn
import torch.nn.functional as F
from .tremor_modules import FrequencyAnalyzer, StatisticalFeatureExtractor

class TremorNetV9(nn.Module):
    def __init__(self, dropout=0.35, all_movements=False, num_movements=11):
        super().__init__()
        self.all_movements = all_movements
        # minimal changes to your multi-scale conv pathways (you can reuse original from V8)
        self.conv_fast = nn.Conv1d(6, 64, kernel_size=3, stride=2, padding=1)
        self.conv_mid = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv_slow = nn.Conv1d(6, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(192)
        self.conv2 = nn.Conv1d(192, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # SE + temporal attention as before
        self.se_fc1 = nn.Linear(128, 16)
        self.se_fc2 = nn.Linear(16, 128)
        self.temporal_attn = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

        # dominant path (kept from V8)
        self.dominant_conv1 = nn.Conv1d(6, 96, 7, stride=2, padding=3)
        self.dominant_bn1 = nn.BatchNorm1d(96)
        self.dominant_conv2 = nn.Conv1d(96, 192, 5, stride=2, padding=2)
        self.dominant_bn2 = nn.BatchNorm1d(192)
        self.dominant_conv3 = nn.Conv1d(192, 256, 3, stride=1, padding=1)
        self.dominant_bn3 = nn.BatchNorm1d(256)
        self.dominant_se1 = nn.Linear(256, 32)
        self.dominant_se2 = nn.Linear(32, 256)

        # NEW Frequency Analyzer and Stats
        self.frequency_analyzer = FrequencyAnalyzer(sample_rate=100, n_fft=256, output_dim=192, dropout=dropout)
        self.stat_extractor = StatisticalFeatureExtractor(out_dim=48)

        # handedness & movement embedding
        self.hand_embed = nn.Embedding(2, 48)
        self.hand_proj = nn.Sequential(nn.Linear(48, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 48))
        if all_movements:
            self.movement_embed = nn.Embedding(num_movements, 32)
            self.movement_proj = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32))

        # contrastive projection
        self.contrast_proj = nn.Linear(128, 64)

        # bilateral attention (reduced dims)
        self.bilateral_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # classifier: update total_dim
        non_dom_dim = 128
        dom_dim = 256
        freq_dim = 192
        stat_dim = 48
        diff_dim = 64
        bilateral_dim = 64
        hand_dim = 48
        movement_dim = 32 if all_movements else 0

        total_dim = non_dom_dim + dom_dim + freq_dim + stat_dim + diff_dim + bilateral_dim + hand_dim + movement_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def _extract_features(self, x):
        fast = self.conv_fast(x)
        mid = self.conv_mid(x)
        slow = self.conv_slow(x)
        x = torch.cat([fast, mid, slow], dim=1)
        x = F.gelu(self.bn1(x))
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        b, c, t = x.shape
        se_pool = x.mean(dim=2)
        se_attn = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool))))
        x = x * se_attn.unsqueeze(2)
        attn_scores = torch.softmax(self.temporal_attn(x.permute(0, 2, 1)), dim=1)
        x_attended = (x.permute(0, 2, 1) * attn_scores).sum(dim=1)
        x_max = x.max(dim=2)[0]
        features = x_attended + 0.3 * x_max
        return features

    def _extract_dominant_features(self, x):
        x = F.gelu(self.dominant_bn1(self.dominant_conv1(x)))
        x = self.dropout(x)
        x = F.gelu(self.dominant_bn2(self.dominant_conv2(x)))
        x = self.dropout(x)
        x = F.gelu(self.dominant_bn3(self.dominant_conv3(x)))
        B, C, T = x.shape
        se = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        se = torch.sigmoid(self.dominant_se2(F.relu(self.dominant_se1(se))))
        x = x * se.unsqueeze(2)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return x

    def forward(self, x, handedness, movements=None, mixup_lambda=None):
        # x: [B, 2, T, 6] -> permute to per-hand [B, 6, T]
        B = x.shape[0]
        left_raw = x[:, 0].permute(0, 2, 1)
        right_raw = x[:, 1].permute(0, 2, 1)
        # handedness embedding
        hand_emb = self.hand_proj(self.hand_embed(handedness.long()))
        # route to dominant/non-dominant (same logic as V8)
        is_left = (handedness == 0).float().view(B, 1, 1)
        is_right = (handedness == 1).float().view(B, 1, 1)
        dominant_raw = is_left * left_raw + is_right * right_raw
        non_dominant_raw = is_right * left_raw + is_left * right_raw

        non_dom_feat = self._extract_features(non_dominant_raw)
        dom_feat = self._extract_dominant_features(dominant_raw)

        # NEW frequency features (bigger)
        freq_feat = self.frequency_analyzer(left_raw, right_raw)

        # stats
        left_stat = self.stat_extractor(left_raw)
        right_stat = self.stat_extractor(right_raw)
        stat_feat = (left_stat + right_stat) / 2.0

        # contrastive
        left_feat = self._extract_features(left_raw)
        right_feat = self._extract_features(right_raw)
        diff_feat = self.contrast_proj(torch.abs(left_feat - right_feat))

        # bilateral attention
        # reuse conv2 pipeline to produce sequences
        left_seq = self.conv2(F.gelu(self.bn1(torch.cat([
            self.conv_fast(left_raw), self.conv_mid(left_raw), self.conv_slow(left_raw)
        ], dim=1))))
        right_seq = self.conv2(F.gelu(self.bn1(torch.cat([
            self.conv_fast(right_raw), self.conv_mid(right_raw), self.conv_slow(right_raw)
        ], dim=1))))
        left_seq = left_seq.permute(0, 2, 1)
        right_seq = right_seq.permute(0, 2, 1)
        bilateral_seq = torch.cat([left_seq, right_seq], dim=1)
        bilateral_attn, _ = self.bilateral_attn(bilateral_seq, bilateral_seq, bilateral_seq)
        bilateral_feat = self.contrast_proj(bilateral_attn.mean(dim=1))

        # combine
        features_list = [non_dom_feat, dom_feat, freq_feat, stat_feat, diff_feat, bilateral_feat, hand_emb]
        if self.all_movements:
            if movements is None:
                raise ValueError("movements required")
            movement_emb = self.movement_proj(self.movement_embed(movements.long()))
            features_list.append(movement_emb)

        combined = torch.cat(features_list, dim=-1)
        logits = self.fusion(combined)
        return logits
