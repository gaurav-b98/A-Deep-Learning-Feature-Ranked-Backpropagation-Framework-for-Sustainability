# Imports
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import init
# Local imports
from .base_model import BaseModel


class ResNetStandard(BaseModel):
    """
    Standard ResNet model with 18 layers.
    Args:
        num_classes (int): Number of classes in the dataset.
    Returns:
        model (nn.Module): ResNet model with 18 layers.
    """
    def __init__(self, num_classes=20):
        super(ResNetStandard, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of ResNet model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            model (nn.Module): ResNet model with 18 layers.
        """
        return self.model(x)


class ResNetSelective(BaseModel):
    """
    ResNet model with selective weight update.
    Args:
        num_classes (int): Number of classes for classification.
        high_percentage (int): Percentage of weights to update.
        low_percentage (int): Percentage of weights to update.
        num_epochs (int): Number of epochs to train.
        schedule (str): Schedule for updating weights (linear or exponential).
    """
    def __init__(self, num_classes=20, high_percentage=30, low_percentage=10,
                 num_epochs=10, schedule='linear'):

        super(ResNetSelective, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self._initialize_weights()
        self.high_percentage = high_percentage / 100.0
        self.low_percentage = low_percentage / 100.0
        self.num_epochs = num_epochs
        self.schedule = schedule

    def forward(self, x):
        """
        Forward pass of ResNet model with selective weight update.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            model (nn.Module): ResNet model with selective weight update.
        """
        return self.model(x)

    def _initialize_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.model.modules():
            # Initialize weights with Kaiming Normal or He Normal
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out',
                                     nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def get_current_percentage(self, epoch):
        """
        Get the current percentage of weights to update.
        Args:
            epoch (int): Current epoch number.
        Returns:
            percentage (float): Current percentage of weights to update.
        """
        # Calculate the percentage of weights to update
        if self.schedule == 'linear':
            percentage = self.high_percentage - (
                (self.high_percentage - self.low_percentage) * (
                    epoch / self.num_epochs))
        elif self.schedule == 'exponential':
            decay_rate = (self.low_percentage / self.high_percentage) ** (
                1 / self.num_epochs)
            percentage = self.high_percentage * (decay_rate ** epoch)
        else:
            # Default to high percentage
            percentage = self.high_percentage
        return percentage

    def apply_selective_weight_update(self, epoch):
        """
        Apply selective weight update based on current percentage
        Args:
            epoch (int): Current epoch
        """
        # Update weights based on dynamic percentage
        current_percentage = self.get_current_percentage(epoch)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Get the top-k percentage of weights to update
                    grad_abs = param.grad.abs().view(-1)
                    k = int(len(grad_abs) * current_percentage)
                    # Skip if no weights to update
                    if k == 0:
                        continue
                    # Compute threshold for kth largest gradient
                    threshold = grad_abs.topk(k, largest=True).values[-1]
                    # Create mask for top gradients
                    mask = (param.grad.abs() >= threshold).float()
                    # Apply mask to gradients
                    param.grad *= mask.view(param.grad.size())
