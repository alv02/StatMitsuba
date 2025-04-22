import torch
import torch.nn.functional as F

imagen = torch.randn(1, 1, 1280, 720)
radius_kernel = 20
pad = radius_kernel * 2
image_padded = F.pad(imagen, (pad, pad, pad, pad))

tile_tam = 256
kernel_size = tile_tam + 2 * radius_kernel
tiles = F.unfold(
    image_padded,
    kernel_size=(kernel_size),
    stride=tile_tam,
)
tiles = tiles.view(1, kernel_size, kernel_size, -1)
print(tiles.shape)
