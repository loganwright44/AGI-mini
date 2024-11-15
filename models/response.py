import torch
import torch.nn as nn
import torch.nn.functional as F


class Response(nn.Module):
  def __init__(
      self,
      conscious_out_features: int,
      unconscious_out_features: int,
      response_out_features: int,
      hidden_features1: int = 1024,
      hidden_features2: int = 512,
      hidden_features3: int = 1024,
      hidden_features4: int = 512,
      hidden_features5: int = 1024
    ):
    super(Response, self).__init__()
    self.activation_layer = nn.ReLU()
    self.fc1 = nn.Linear(in_features=conscious_out_features + unconscious_out_features, out_features=hidden_features1)
    self.fc2 = nn.Linear(in_features=hidden_features1, out_features=hidden_features2)
    self.fc3 = nn.Linear(in_features=hidden_features2, out_features=hidden_features3)
    self.fc4 = nn.Linear(in_features=hidden_features3, out_features=hidden_features4)
    self.fc5 = nn.Linear(in_features=hidden_features4, out_features=hidden_features5)
    self.fc6 = nn.Linear(in_features=hidden_features5, out_features=response_out_features)
  
  def forward(self, conscious_out_vector: torch.Tensor, unconscious_out_vector: torch.Tensor) -> torch.Tensor:
    input_vector = torch.cat((conscious_out_vector, unconscious_out_vector), dim=-1)
    x = self.fc1(input_vector)
    x = self.activation_layer(x)
    x = self.fc2(x)
    x = self.activation_layer(x)
    x = self.fc3(x)
    x = self.activation_layer(x)
    x = self.fc4(x)
    x = self.activation_layer(x)
    x = self.fc5(x)
    x = self.activation_layer(x)
    x = self.fc6(x)
    x = self.activation_layer(x)
    return x

