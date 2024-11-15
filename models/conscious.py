import torch
import torch.nn as nn
import torch.nn.functional as F


class Conscious(nn.Module):
  def __init__(self, state_vector_features: int, embedding_dim: int, out_features: int, device: torch.device, hidden_features: int = 2048):
    super(Conscious, self).__init__()
    self.embedding = nn.Linear(in_features=state_vector_features + out_features, out_features=embedding_dim, device=device)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embedding_dim,
      dim_feedforward=hidden_features,
      nhead=8,
      batch_first=True,
      device=device
    )
    self.encoder = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=12
    )
    self.fc_out = nn.Linear(in_features=hidden_features, out_features=out_features, device=device)
    self.out_features = out_features
  
  def forward(self, state_vector: torch.Tensor, conscious_out_vector: torch.Tensor) -> torch.Tensor:
    input_vector = torch.cat((conscious_out_vector, state_vector), dim=-1)
    x = self.embedding(input_vector)
    x = self.encoder(x)
    x = self.fc_out(x)
    return x

