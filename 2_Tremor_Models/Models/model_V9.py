"""
TremorNetV9 - Enhanced with Clinical Metadata Integration

Combines:
1. Multi-scale CNN (fast/mid/slow tremor frequencies)
2. Squeeze-excitation channel attention
3. Temporal attention
4. Dominant hand 3x capacity pathway
5. FFT frequency analysis (power + phase + envelope)
6. Statistical moments
7. Contrastive features (asymmetry)
8. Bilateral coordination
9. Optional movement type embedding
10. **NEW: Clinical metadata integration with cross-attention**
    - Age-specific tremor patterns
    - Body measurements (height/weight/BMI) correlation with tremor amplitude
    - Family history features
    - Alcohol effect indicators
"""
import torch
import torch.nn.functional as F
from torch import nn
from .tremor_modules import FrequencyAnalyzer, StatisticalFeatureExtractor, ClinicalMetadataEncoder

class TremorNetV9(nn.Module):
    """
    Compressed TremorNetV9 with clinical metadata cross-attention.
    Behavior, shapes and module outputs are preserved.
    """
    def __init__(self, dropout=0.45, all_movements=False, num_movements=11):
        super().__init__()
        self.all_movements = all_movements
        self.dropout = nn.Dropout(dropout)

        # ------------- multi-scale time-domain convs (shared shapes) -------------
        self.conv_fast = nn.Conv1d(6, 64, 3, stride=2, padding=1) # short kernel
        self.conv_mid  = nn.Conv1d(6, 64, 7, stride=2, padding=3)
        self.conv_slow = nn.Conv1d(6, 64, 15, stride=2, padding=7) # long kernel
        self.bn1 = nn.BatchNorm1d(192)
        self.conv2 = nn.Conv1d(192, 128, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # SE (squeeze-excite) to highlight important channels + temporal attn (picks the most useful time steps)
        self.se_fc1, self.se_fc2 = nn.Linear(128, 16), nn.Linear(16, 128)
        self.temporal_attn = nn.Linear(128, 1)

        # ------------- dominant-hand super-pathway (3x capacity) -------------
        self.dom_conv1 = nn.Conv1d(6, 96, 7, stride=2, padding=3); self.dom_bn1 = nn.BatchNorm1d(96)
        self.dom_conv2 = nn.Conv1d(96, 192, 5, stride=2, padding=2); self.dom_bn2 = nn.BatchNorm1d(192)
        self.dom_conv3 = nn.Conv1d(192, 256, 3, stride=1, padding=1); self.dom_bn3 = nn.BatchNorm1d(256)
        self.dom_se1, self.dom_se2 = nn.Linear(256, 32), nn.Linear(32, 256)

        # ------------- frequency & statistics -------------
        self.frequency_analyzer = FrequencyAnalyzer(output_dim=128, dropout=dropout)
        self.stat_extractor = StatisticalFeatureExtractor()

        # ------------- embeddings -------------
        self.hand_embed = nn.Embedding(2, 48)
        self.hand_proj = nn.Sequential(nn.Linear(48, 96), nn.LayerNorm(96), nn.Tanh(),
                                       nn.Dropout(dropout * 0.3), nn.Linear(96, 48))
        if all_movements:
            self.movement_embed = nn.Embedding(num_movements, 32)
            self.movement_proj  = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64),
                                                nn.GELU(), nn.Dropout(dropout * 0.3), nn.Linear(64, 32))

        # ------------- clinical metadata + attention -------------
        self.metadata_encoder = ClinicalMetadataEncoder(output_dim=96, dropout=dropout)
        self.metadata_to_signal = nn.Linear(96, 128)            # learnable fix for previous bug
        self.signal_metadata_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4,
                                                          dropout=dropout * 0.5, batch_first=True)

        # bilateral coordination attention
        self.bilateral_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4,
                                                    dropout=dropout * 0.5, batch_first=True)

        # projections
        self.contrast_proj = nn.Linear(128, 64)
        self.dom_proj = nn.Linear(256, 128)

        # ------------- fusion/classifier (keeps original dims) -------------
        non_dom_dim, dom_dim, freq_dim = 128, 256, 128
        stat_dim, diff_dim, bilateral_dim, hand_dim, metadata_dim = 32, 64, 64, 48, 96
        movement_dim = 32 if all_movements else 0
        total_dim = non_dom_dim + dom_dim + freq_dim + stat_dim + diff_dim + bilateral_dim + hand_dim + movement_dim + metadata_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 640), nn.BatchNorm1d(640), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(640, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 320), nn.BatchNorm1d(320), nn.GELU(), nn.Dropout(dropout * 0.7),
            nn.Linear(320, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    # -------------------- helpers --------------------
    def _mixup_apply(self, B, x, metadata, handedness, movements, mixup_lambda):
        """Return (x, metadata, hand_emb_weights, movements_onehot) after mixup (if applied)."""
        if mixup_lambda is None or not self.training:
            return x, metadata, None, None
        idx = torch.randperm(B, device=x.device)
        x = mixup_lambda * x + (1 - mixup_lambda) * x[idx]
        metadata = mixup_lambda * metadata + (1 - mixup_lambda) * metadata[idx]

        hand_oh = F.one_hot(handedness.long(), 2).float()
        hand_mix = mixup_lambda * hand_oh + (1 - mixup_lambda) * hand_oh[idx]

        move_mix = None
        if self.all_movements and movements is not None:
            move_oh = F.one_hot(movements.long(), self.movement_embed.num_embeddings).float()
            move_mix = mixup_lambda * move_oh + (1 - mixup_lambda) * move_oh[idx]

        return x, metadata, hand_mix, move_mix

    def _ms_features(self, x):
        """Multi-scale CNN → SE → temporal attention → aggregated [B,128]. x: [B,6,T]"""
        f = self.conv_fast(x); m = self.conv_mid(x); s = self.conv_slow(x)
        x_cat = torch.cat([f, m, s], dim=1)
        x_cat = F.gelu(self.bn1(x_cat)); x_cat = self.dropout(x_cat)
        x_cat = F.gelu(self.bn2(self.conv2(x_cat)))  # [B,128,T']

        se_pool = x_cat.mean(dim=2)
        se_attn = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool))))
        x_se = x_cat * se_attn.unsqueeze(2)

        attn_scores = torch.softmax(self.temporal_attn(x_se.permute(0, 2, 1)), dim=1)
        x_att = (x_se.permute(0, 2, 1) * attn_scores).sum(dim=1)
        x_max = x_se.max(dim=2)[0]
        return x_att + 0.3 * x_max

    def _dominant_features(self, x):
        """Dominant super-pathway → SE → pooled [B,256]"""
        x = F.gelu(self.dom_bn1(self.dom_conv1(x))); x = self.dropout(x)
        x = F.gelu(self.dom_bn2(self.dom_conv2(x))); x = self.dropout(x)
        x = F.gelu(self.dom_bn3(self.dom_conv3(x)))

        se = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        se_w = torch.sigmoid(self.dom_se2(F.relu(self.dom_se1(se))))
        x = x * se_w.unsqueeze(2)
        return F.adaptive_avg_pool1d(x, 1).squeeze(2)

    # -------------------- forward --------------------
    def forward(self, x, handedness, metadata, movements=None, mixup_lambda=None):
        """
        x: [B,2,T,6]; handedness: [B]; metadata: [B, ...]; movements: [B] (optional)
        """
        B = x.shape[0]

        # apply mixup (if requested)
        x, metadata, handedness_mix, movements_mix = self._mixup_apply(B, x, metadata, handedness, movements, mixup_lambda)

        # left/right raw signals as [B,6,T]
        left_raw  = x[:, 0].permute(0, 2, 1)
        right_raw = x[:, 1].permute(0, 2, 1)

        # handedness embedding (supports mixup weights)
        if handedness_mix is not None:
            hand_emb = handedness_mix.to(x.device) @ self.hand_embed.weight
        else:
            hand_emb = self.hand_embed(handedness.long())
        hand_emb = self.hand_proj(hand_emb)

        # movement embedding (optional)
        if self.all_movements:
            if movements is None and movements_mix is None:
                raise ValueError("movements required when all_movements=True")
            if movements_mix is not None:
                move_emb = movements_mix.to(x.device) @ self.movement_embed.weight
            else:
                move_emb = self.movement_embed(movements.long())
            move_emb = self.movement_proj(move_emb)
            move_emb = F.normalize(move_emb, dim=-1)

        # metadata encoding (learnable) -> [B,96]
        metadata_feat = self.metadata_encoder(metadata)

        # dominant/non-dominant routing (supports mixup weights)
        if mixup_lambda is not None and self.training and handedness_mix is not None:
            lw = handedness_mix[:, 0].view(B, 1, 1).to(x.device); rw = handedness_mix[:, 1].view(B, 1, 1).to(x.device)
            dominant_raw = lw * left_raw + rw * right_raw
            non_dominant_raw = rw * left_raw + lw * right_raw
        else:
            is_left  = (handedness == 0).float().view(B, 1, 1)
            is_right = (handedness == 1).float().view(B, 1, 1)
            dominant_raw = is_left * left_raw + is_right * right_raw
            non_dominant_raw = is_right * left_raw + is_left * right_raw

        # feature extraction
        non_dom_feat = self._ms_features(non_dominant_raw)     # [B,128]
        dom_feat = self._dominant_features(dominant_raw)      # [B,256]

        # frequency + stats
        freq_feat = self.frequency_analyzer(left_raw, right_raw)  # [B,128]
        left_stat = self.stat_extractor(left_raw); right_stat = self.stat_extractor(right_raw)
        stat_feat = (left_stat + right_stat) / 2                   # [B,32]

        # contrastive asymmetry
        left_feat  = self._ms_features(left_raw)
        right_feat = self._ms_features(right_raw)
        diff_feat = self.contrast_proj(torch.abs(left_feat - right_feat))  # [B,64]

        # bilateral coordination via attention over sequences
        # reuse shared multi-scale conv -> conv2 path for left/right sequences
        def seq_from_raw(raw):
            f = self.conv_fast(raw); m = self.conv_mid(raw); s = self.conv_slow(raw)
            seq = self.conv2(F.gelu(self.bn1(torch.cat([f, m, s], dim=1))))
            return seq.permute(0, 2, 1)  # [B, SeqLen, 128]
        left_seq, right_seq = seq_from_raw(left_raw), seq_from_raw(right_raw)
        bilateral_seq = torch.cat([left_seq, right_seq], dim=1)
        bilateral_attn, _ = self.bilateral_attn(bilateral_seq, bilateral_seq, bilateral_seq)
        bilateral_feat = self.contrast_proj(bilateral_attn.mean(dim=1))  # [B,64]

        # metadata-modulated signal features (cross-attention)
        dom_feat_proj = self.dom_proj(dom_feat)           # [B,128]
        signal_stack = torch.stack([non_dom_feat, dom_feat_proj, freq_feat], dim=1)  # [B,3,128]

        metadata_proj = self.metadata_to_signal(metadata_feat).unsqueeze(1)  # [B,1,128] (learnable)
        modulated_signal, _ = self.signal_metadata_attn(metadata_proj, signal_stack, signal_stack)
        modulated_signal = modulated_signal.squeeze(1)
        non_dom_feat = non_dom_feat + 0.2 * modulated_signal

        # assemble features
        feat_list = [non_dom_feat, dom_feat, freq_feat, stat_feat, diff_feat, bilateral_feat, hand_emb, metadata_feat]
        if self.all_movements:
            feat_list.append(move_emb)
        combined = torch.cat(feat_list, dim=-1)

        return self.fusion(combined)
