import math

import torch
import torch.nn.functional as F


def extract_patches_3ds(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    channels = x.shape[0]

    x = torch.nn.functional.pad(x, padding)
    # (C, H, W, T)
    x = (
        x.unfold(1, kernel_size[0], stride[0])
        .unfold(2, kernel_size[1], stride[1])
        .unfold(3, kernel_size[2], stride[2])
    )
    # (C, h_dim_out, w_dim_out, t_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.contiguous().view(
        channels, -1, kernel_size[0], kernel_size[1], kernel_size[2]
    )
    # (C, h_dim_out * w_dim_out * t_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    print(x.shape)
    x = x.permute(0, 2, 3, 4, 1)
    x = x.contiguous().view(
        channels * kernel_size[0] * kernel_size[1] * kernel_size[2], -1
    )
    # (C* kernel_size[0] * kernel_size[1] * kernel_size[2]m h_dim_out * w_dim_out * t_dim_out)
    print(x.shape)
    return x


def combine_patches_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(
        d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    h_dim_in = get_dim_blocks(
        h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    w_dim_in = get_dim_blocks(
        w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(
        -1,
        channels,
        d_dim_in,
        h_dim_in,
        w_dim_in,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(
        -1,
        channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2],
        h_dim_in * w_dim_in,
    )
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(
        x,
        output_size=(h_dim_out, w_dim_out),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(
        x,
        output_size=(d_dim_out, h_dim_out * w_dim_out),
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x


radius = 5
final_tile_size = 32
tile_size = final_tile_size + 2 * radius
stride = final_tile_size
a = torch.arange(1, 57600001, dtype=torch.float).view(3, 400, 400, 120)
a_padded = F.pad(
    a,
    (
        radius,  # T left
        final_tile_size - 1 + radius,  # T right
        radius,  # W left
        final_tile_size - 1 + radius,  # W right
        radius,  # H left
        final_tile_size - 1 + radius,  # H right
    ),
)


print(a_padded.shape)
print(a_padded[:, 23, 24, 21])
b = extract_patches_3ds(a_padded, kernel_size=tile_size, padding=0, stride=stride)
b = b.view(3, tile_size, tile_size, tile_size, -1)
b = b.permute(4, 0, 1, 2, 3)
b = b.reshape(-1, 3, tile_size, tile_size, tile_size)
print(b[0, :, 23, 24, 21])
print(b.shape)

H_OUT = math.ceil(400 / final_tile_size) * final_tile_size
W_OUT = math.ceil(400 / final_tile_size) * final_tile_size
T_OUT = math.ceil(120 / final_tile_size) * final_tile_size
print(H_OUT)
print(T_OUT)
c = combine_patches_3d(
    b,
    kernel_size=tile_size,
    output_shape=(1, 3, 441, 441, 161),
    padding=0,
    stride=stride,
)
print(c.shape)
