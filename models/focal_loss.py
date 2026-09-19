import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    Formula: Loss = -alpha * (1 - p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Optional: can be a tensor of weights for each class
        self.reduction = reduction

    def forward(self, inputs, targets, gamma=None):
        """
        inputs: (B, C) logits
        targets: (B) labels
        gamma: (float) optional override for gamma
        """
        current_gamma = gamma if gamma is not None else self.gamma
        
        # Use log_softmax for numerical stability
        logpt = F.log_softmax(inputs, dim=1)
        
        # Gather the log probabilities of the target class
        logpt = logpt.gather(1, targets.view(-1, 1))
        logpt = logpt.view(-1)
        
        pt = logpt.exp()
        
        # Critical fix: Clamp pt to avoid numerical instability when pt is close to 1
        # This prevents NaN gradients when (1-pt) becomes 0 and gamma < 1 during backprop
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        ce_loss = -logpt
        focal_loss = (1 - pt) ** current_gamma * ce_loss
        
        if self.alpha is not None:
            # Handle alpha weighting if provided
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            else:
                # Assuming alpha is a tensor of size C
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
