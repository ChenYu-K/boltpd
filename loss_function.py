import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        residual = torch.abs(y_true - y_pred)
        condition = residual < self.delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = self.delta * (residual - 0.5 * self.delta)
        loss = torch.where(condition, squared_loss, linear_loss)
        return torch.mean(loss)
