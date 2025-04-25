import torch
import torch.nn.functional as F
from denoiserVectorized import Tile

imagen = torch.randn(1, 1, 1281, 720)
tile = Tile(5)
result = tile(imagen)
