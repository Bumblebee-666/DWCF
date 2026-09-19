import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-Modal Contrastive Loss (InfoNCE style)
    Aligns audio and visual features by pulling positive pairs together 
    and pushing negative pairs apart.
    """
    def __init__(self, temperature=0.07):
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_features, visual_features):
        """
        Args:
            audio_features: (batch_size, feature_dim)
            visual_features: (batch_size, feature_dim)
        """
        # Normalize features along the feature dimension
        audio_features = F.normalize(audio_features, dim=1)
        visual_features = F.normalize(visual_features, dim=1)
        
        # Compute similarity matrix (scaled dot product)
        # (B, D) @ (D, B) -> (B, B)
        logits = torch.matmul(audio_features, visual_features.T) / self.temperature
        
        # Labels: the diagonal elements are the positive pairs
        # labels[i] = i means the i-th audio matches the i-th visual
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # Symmetric loss: Audio->Visual and Visual->Audio
        loss_a2v = F.cross_entropy(logits, labels)
        loss_v2a = F.cross_entropy(logits.T, labels)
        
        return (loss_a2v + loss_v2a) / 2
