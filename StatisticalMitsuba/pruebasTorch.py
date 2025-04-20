import torch
import torch.nn.functional as F

imagen = torch.randn(1, 3, 10, 10)
vecinos = F.unfold(imagen, kernel_size=(3, 3), padding=1)
print(vecinos.shape)
