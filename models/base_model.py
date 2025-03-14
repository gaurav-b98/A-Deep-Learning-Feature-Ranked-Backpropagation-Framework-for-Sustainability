# Imports
import torch.nn as nn
import torch


class BaseModel(nn.Module):
    """
    Base model class for all models
    Args:
        nn.Module: PyTorch module
    Returns:
        model (nn.Module): Base model class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def compute_saliency_maps(self, inputs, labels):
        """
        Compute saliency maps for input samples
        Args:
            inputs (torch.Tensor): Input samples
            labels (torch.Tensor): Labels for input samples
        Returns:
            saliency (torch.Tensor): Saliency maps for input samples
        """
        # Set model in evaluation mode
        self.eval()

        # Set requires_grad attribute of inputs to True
        inputs.requires_grad_()

        # Forward pass
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass
        loss.backward()

        # Compute saliency maps
        saliency = inputs.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)

        return saliency
