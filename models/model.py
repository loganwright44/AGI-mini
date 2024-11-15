import torch


class Model(torch.nn.Module):
  def __init__(self, in_features: int, device: torch.device):
    super(Model, self).__init__()
    self.fc = torch.nn.Linear(in_features=10, out_features=1)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)

