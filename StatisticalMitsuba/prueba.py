# Joint Bilateral Filter with Membership Functions in PyTorch
import time

import matplotlib

matplotlib.use("Agg")
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


def extract_patches_3d(x, kernel_size, padding=0, stride=1):
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
    x = x.permute(0, 2, 3, 4, 1)
    x = x.contiguous().view(
        channels, kernel_size[0] * kernel_size[1] * kernel_size[2], -1
    )
    # (C, kernel_size[0] * kernel_size[1] * kernel_size[2]m h_dim_out * w_dim_out * t_dim_out)
    return x


def extract_tiles_3d(x, kernel_size, padding=0, stride=1):
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
    x = x.permute(1, 2, 3, 0, 4, 5, 6).reshape(
        -1, channels, kernel_size[0], kernel_size[1], kernel_size[2]
    )
    # x = x.contiguous().view(
    #     -1, channels, kernel_size[0], kernel_size[1], kernel_size[2]
    # )
    # (h_dim_out * w_dim_out * t_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    return x


def combine_tiles_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
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
    h_dim_out, w_dim_out, t_dim_out = output_shape[2:]
    h_dim_inb = get_dim_blocks(
        h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    w_dim_inb = get_dim_blocks(
        w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    t_dim_in = get_dim_blocks(
        t_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    x.permute(1, 0, 2, 3, 4)

    x = x.view(
        channels,
        h_dim_inb,
        w_dim_inb,
        t_dim_in,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    # (C, h_dim_in, w_dim_in, t_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 4, 1, 5, 6, 2, 3)
    # (C, kernel_size[0], h_dim_in, kernel_size[1], kernel_size[2], w_dim_in, t_dim_in)

    x = x.contiguous().view(
        -1,
        channels * kernel_size[0] * h_dim_inb * kernel_size[1] * kernel_size[2],
        w_dim_inb * t_dim_in,
    )
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(
        x,
        output_size=(w_dim_out, t_dim_out),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], h_dim_inb * w_dim_out * t_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(
        x,
        output_size=(h_dim_out, w_dim_out * t_dim_out),
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C, D, H * W)

    x = x.view(-1, channels, h_dim_out, w_dim_out, t_dim_out)
    # (B, C, D, H, W)

    return x


class Tile(nn.Module):
    """
    Creates a tiled tensor for and image
    """

    def __init__(self, spatial_radius, temporal_radius):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.final_spatial_tile_size = 200
        self.final_temporal_tile_size = 100
        self.spatial_tile_size = self.final_spatial_tile_size + 2 * self.spatial_radius
        self.temporal_tile_size = (
            self.final_temporal_tile_size + 2 * self.temporal_radius
        )
        self.stride = (
            self.final_spatial_tile_size,
            self.final_spatial_tile_size,
            self.final_temporal_tile_size,
        )
        self.kernel_size = (
            self.spatial_tile_size,
            self.spatial_tile_size,
            self.temporal_tile_size,
        )

    def forward(self, x):
        """
        Returns tensor (tiles, C, H, W, T)
        """
        c, h, w, t = x.shape
        x_padded = F.pad(
            x,
            (
                self.temporal_radius,  # T left
                self.final_temporal_tile_size - 1 + self.temporal_radius,  # T right
                self.spatial_radius,  # W left
                self.final_spatial_tile_size - 1 + self.spatial_radius,  # W right
                self.spatial_radius,  # H left
                self.final_spatial_tile_size - 1 + self.spatial_radius,  # H right
            ),
        )
        padding = (
            self.temporal_radius,  # T left
            self.final_temporal_tile_size - 1 + self.temporal_radius,  # T right
            self.spatial_radius,  # W left
            self.final_spatial_tile_size - 1 + self.spatial_radius,  # W right
            self.spatial_radius,  # H left
            self.final_spatial_tile_size - 1 + self.spatial_radius,
        )  # H right

        np.save(
            "./x_padded.npy",
            x_padded.permute(1, 2, 3, 0).cpu().numpy(),
        )
        print(x_padded.shape)
        tiles = extract_tiles_3d(
            x=x, kernel_size=self.kernel_size, padding=padding, stride=self.stride
        )

        return tiles


class Shift(nn.Module):
    """
    Creates a tensor with the neighbours of each pixel
    """

    def __init__(self, spatial_radius, temporal_radius):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.spatial_kernel_size = 2 * self.spatial_radius + 1
        self.temporal_kernel_size = 2 * self.temporal_radius + 1
        self.n_patches = self.spatial_kernel_size**2 * self.temporal_radius
        self.kernel_size = (
            self.spatial_kernel_size,
            self.spatial_kernel_size,
            self.temporal_kernel_size,
        )

    def forward(self, x):
        """
        x (C, H, W, T)
        returns (C, n_patches, H_out * W_out * T_out)
        where H_out = H - 2*radius, W_out = W - 2*radius, T_out = T - 2*radius
        """
        c, h, w, t = x.shape
        patches = extract_patches_3d(x, kernel_size=self.kernel_size)

        return patches


def load_denoiser_data(
    scene_path: str, stats_path: str, transient_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Carga todos los datos necesarios para el denoiser: g-buffers, estadísticas y video transiente.

    Args:
        scene_path: Ruta al archivo .exr con los g-buffers
        stats_path: Ruta al archivo .npy con las estadísticas
        transient_path: Ruta al archivo .npy con el video transiente

    Returns:
        Tuple con: guidance, estimands, estimands_variance, images, spp, shape
        Todos los tensores en formato [C, H, W, T]
        - guidance: 9 canales (pos_x, pos_y, albedo_r, albedo_g, albedo_b, normal_x, normal_y, normal_z, temporal)
    """
    # Load transient video
    images = np.load(transient_path)
    images = (
        torch.from_numpy(images).to(torch.float32).permute(3, 0, 1, 2)
    )  # [C, H, W, T]

    _, h, w, t = images.shape

    # Load G-Buffers
    bitmap = mi.Bitmap(scene_path + ".exr")
    res = dict(bitmap.split())

    # Load albed and normals from the bitmap
    albedo = (
        torch.from_numpy(np.array(res["albedo"], dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    normals = (
        torch.from_numpy(np.array(res["nn"], dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    # TODO: Mejorar esto usando torch funcions en vez de for
    # Generar features de posición
    y_coords = torch.linspace(-1, 1, h, dtype=torch.float32)
    x_coords = torch.linspace(-w / h, w / h, w, dtype=torch.float32)
    z_coords = torch.linspace(0, 1, t, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Crear pos con dimensión temporal [3, H, W, T]
    pos_list = []
    for i in range(t):
        z_val = z_coords[i]
        z_grid = torch.full_like(x_grid, z_val)
        pos_frame = torch.stack([x_grid, y_grid, z_grid], dim=0)  # [3, H, W]
        pos_list.append(pos_frame)
    pos = torch.stack(pos_list, dim=-1)  # [3, H, W, T]

    # Expandir albedo y normals para la dimensión temporal
    albedo = albedo.squeeze(0).unsqueeze(-1).expand(-1, -1, -1, t)  # [3, H, W, T]
    normals = normals.squeeze(0).unsqueeze(-1).expand(-1, -1, -1, t)  # [3, H, W, T]

    # Concatenar guidance features
    guidance = torch.cat(
        [pos, albedo, normals], dim=0
    )  # [9, H, W, T]    # Load statistics
    statistics = np.load(stats_path)  # [H, W, T, C, 3]
    estimands = (
        torch.from_numpy(statistics[..., 0])
        .to(torch.float32)
        .permute(3, 0, 1, 2)  # [C, H, W, T]
    )
    estimands_variance = (
        torch.from_numpy(statistics[..., 1])
        .to(torch.float32)
        .permute(3, 0, 1, 2)  # [C, H, W, T]
    )
    spp = statistics[0, 0, 0, 0, 2]
    return guidance, estimands, estimands_variance, images, spp


if __name__ == "__main__":
    # Configuración de rutas
    scene_path = "./io/cbox/imagen"
    stats_path = "./io/transient/transient_stats_2048.npy"
    transient_path = "./io/transient/transient_data_2048.npy"

    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    # Cargar todos los datos
    guidance, estimands, estimands_variance, images, spp = load_denoiser_data(
        scene_path, stats_path, transient_path
    )

    tiler = Tile(5, 0)
    images_tiled = tiler(images)
    print(images_tiled.shape)
    np.save(
        "./debug_output/original_tile.npy",
        images_tiled[0].permute(1, 2, 3, 0).cpu().numpy(),
    )
