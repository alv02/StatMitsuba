# Joint Bilateral Filter with Membership Functions in PyTorch
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import drjit as dr
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
    x = x.permute(0, 4, 5, 6, 1, 2, 3).reshape(
        channels, kernel_size[0] * kernel_size[1] * kernel_size[2], -1
    )

    # (C, kernel_size[0] * kernel_size[1] * kernel_size[2],h_dim_out * w_dim_out * t_dim_out)
    return x


def get_dim_blocks(dim_in, kernel_size, padding=0, stride=1, dilation=1):
    return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def extract_tiles_3d(x, kernel_size, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    x = x.contiguous()

    channels, depth, height, width = x.shape[-4:]
    d_blocks = get_dim_blocks(
        depth, kernel_size=kernel_size[0], stride=stride[0], dilation=dilation[0]
    )
    h_blocks = get_dim_blocks(
        height, kernel_size=kernel_size[1], stride=stride[1], dilation=dilation[1]
    )
    w_blocks = get_dim_blocks(
        width, kernel_size=kernel_size[2], stride=stride[2], dilation=dilation[2]
    )
    shape = (
        channels,
        d_blocks,
        h_blocks,
        w_blocks,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    strides = (
        width * height * depth,
        stride[0] * width * height,
        stride[1] * width,
        stride[2],
        dilation[0] * width * height,
        dilation[1] * width,
        dilation[2],
    )

    x = x.as_strided(shape, strides)
    x = x.permute(1, 2, 3, 0, 4, 5, 6)
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

    x = x.permute(1, 0, 2, 3, 4)
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
        self.final_spatial_tile_size = 16
        self.final_temporal_tile_size = 16
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

    def forward(self, x, padding_value=0):
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
            mode="constant",
            value=padding_value,
        )

        tiles = extract_tiles_3d(
            x=x_padded, kernel_size=self.kernel_size, stride=self.stride
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


class StatDenoiser(nn.Module):
    """
    Joint Bilateral Filter with Membership Functions uses guidance image to calculate weights
    and statistical tests to determine which pixels should be combined.
    """

    def __init__(
        self,
        spatial_radius=5,
        temporal_radius=5,
        alpha=0.005,
        spp=0,
        debug_pixels=None,
    ):
        super(StatDenoiser, self).__init__()

        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.spatial_kernel_size = 2 * self.spatial_radius + 1
        self.temporal_kernel_size = 2 * self.temporal_radius + 1
        self.alpha = alpha
        self.n_patches = self.spatial_kernel_size**2 * self.temporal_kernel_size

        # Debug parameters

        # Sigma_inv(1, C*n_patches, 1, 1)
        sigma_inv = torch.tensor(
            [0.1, 0.1, 0.1, 50, 50, 50, 10, 10, 10], dtype=torch.float32
        )
        sigma_inv = torch.reshape(sigma_inv, (-1, 1, 1))

        self.register_buffer("sigma_inv", sigma_inv)
        # Create shift operator
        self.shift = Shift(
            spatial_radius=spatial_radius, temporal_radius=temporal_radius
        )
        self.tile = Tile(spatial_radius=spatial_radius, temporal_radius=temporal_radius)
        self.debug_pixels = debug_pixels

    def get_debug_pixels_tiled(self, debug_pixels, H, W, T):
        tiled_debug_pixels = []

        tiles_y = math.ceil(H / self.tile.final_spatial_tile_size)
        tiles_x = math.ceil(W / self.tile.final_spatial_tile_size)
        tiles_t = math.ceil(T / self.tile.final_temporal_tile_size)

        for pixel_x, pixel_y, pixel_t in debug_pixels:
            tile_y = pixel_y // tiles_y
            tile_x = pixel_x // tiles_x
            tile_t = pixel_t // tiles_t

            index = dr.fma(tile_y, tiles_x, tile_x)
            index = dr.fma(index, tiles_t, tile_t)

            offset_y = pixel_y % self.tile.final_spatial_tile_size
            offset_x = pixel_x % self.tile.final_spatial_tile_size
            offset_t = pixel_t % self.tile.final_temporal_tile_size

            tiled_debug_pixels.append(
                (index, offset_x, offset_y, offset_t, pixel_x, pixel_y, pixel_t)
            )

        return tiled_debug_pixels

    def debug(
        self,
        original_tile,
        current_tile_index,
        H,
        W,
        T,
    ):
        if debug_pixels == None:
            return
        debug_pixels_tiled = self.get_debug_pixels_tiled(self.debug_pixels, H, W, T)

        for (
            tile_index,
            offset_x,
            offset_y,
            offset_t,
            pixel_x,
            pixel_y,
            pixel_t,
        ) in debug_pixels_tiled:
            if tile_index == current_tile_index:
                print(tile_index)
                print(original_tile.shape)
                original_tile = original_tile.reshape(
                    3,
                    -1,
                    self.tile.final_spatial_tile_size,
                    self.tile.final_spatial_tile_size,
                    self.tile.final_temporal_tile_size,
                )

                original = (
                    original_tile[:, :, offset_y, offset_x, offset_t]
                    .cpu()
                    .numpy()
                    .reshape(
                        self.spatial_kernel_size,
                        self.spatial_kernel_size,
                        self.temporal_kernel_size,
                        -1,
                    )
                )

                np.save(
                    f"./debug_output/transient/patches{pixel_x}_{pixel_y}_{pixel_t}.npy",
                    original,
                )

    def forward(self, images, guidance, estimands, estimands_variance, spp):
        # Tile inputs
        tiled_images = self.tile(images)
        tiled_guidance = self.tile(guidance)
        tiled_estimands = self.tile(estimands, -2)
        tiled_estimands_variance = self.tile(estimands_variance)
        C, H, W, T = images.shape

        # Procesamiento por tile
        tiles_y, tiles_x, tiles_t, _, _, _, _ = tiled_images.shape
        tiled_denoised_image = torch.empty(
            (
                tiles_y * tiles_x * tiles_t,
                C,
                self.tile.final_spatial_tile_size,
                self.tile.final_spatial_tile_size,
                self.tile.final_temporal_tile_size,
            )
        )

        np.save("tile.npy", tiled_denoised_image[10, 12, 3, :].reshape())


def load_denoiser_data(
    aovs_path: str, stats_path: str, transient_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
        torch.from_numpy(images).to(torch.float32).permute(3, 0, 1, 2).contiguous()
    )  # [C, H, W, T]

    _, h, w, t = images.shape

    # Load G-Buffers
    bitmap = mi.Bitmap(aovs_path)
    res = dict(bitmap.split())

    # Load albed and normals from the bitmap
    albedo = (
        torch.from_numpy(np.array(res["albedo"], dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    normals = (
        torch.from_numpy(np.array(res["nn"], dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )

    # TODO: Mejorar esto usando torch funcions en vez de for
    # Generar features de posición
    y_coords = torch.linspace(-1, 1, h, dtype=torch.float32)
    x_coords = torch.linspace(-w / h, w / h, w, dtype=torch.float32)
    z_coords = torch.linspace(-1, 1, t, dtype=torch.float32)
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
        .permute(3, 0, 1, 2)
        .contiguous()  # [C, H, W, T]
    )
    estimands_variance = (
        torch.from_numpy(statistics[..., 1])
        .to(torch.float32)
        .permute(3, 0, 1, 2)
        .contiguous()  # [C, H, W, T]
    )
    spp = int(statistics[0, 0, 0, 0, 2])
    return guidance, estimands, estimands_variance, images, spp


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Uso: python denoiserTransient.py <escena> <spp> <spatial_radius> <temporal_radius> <alpha>"
        )
        print("Ejemplo: python denoiserTransient.py staircase 2048 8 3 0.05")
        sys.exit(1)

    escena = sys.argv[1]
    spp = int(sys.argv[2])
    spatial_radius = int(sys.argv[3])
    temporal_radius = int(sys.argv[4])
    alpha = float(sys.argv[5])

    # Configuración de paths basada en escena y spp
    base_io = "./io"
    aovs_path = f"{base_io}/steady/{escena}/imagen.exr"
    stats_path = f"{base_io}/transient/{escena}/transient_stats_{spp}.npy"
    transient_path = f"{base_io}/transient/{escena}/transient_data_{spp}.npy"

    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    # Cargar todos los datos
    guidance, estimands, estimands_variance, images, spp = load_denoiser_data(
        aovs_path, stats_path, transient_path
    )

    # TODO: Esto es para debug en algun momento habrá que quitarlo
    # Guardar estimands
    np.save(
        f"{base_io}/transient/{escena}/estimands_{spp}.npy",
        estimands.permute(1, 2, 3, 0),
    )
    np.save(
        f"{base_io}/transient/{escena}/estimands_variance_{spp}.npy",
        estimands_variance.permute(1, 2, 3, 0),
    )

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mover datos al dispositivo
    guidance = guidance.to(device)
    estimands = estimands.to(device)
    estimands_variance = estimands_variance.to(device)
    images = images.to(device)

    # Configurar denoiser
    debug_pixels = [(1085, 615, 0)]
    stat_denoiser = StatDenoiser(
        spatial_radius=spatial_radius,
        temporal_radius=temporal_radius,
        alpha=alpha,
        spp=spp,
        debug_pixels=debug_pixels,
    )
    stat_denoiser = stat_denoiser.to(device)

    # Aplicar denoising
    print("Aplicando denoising...")
    start_time = time.time()
    with torch.no_grad():
        result = stat_denoiser(
            images,
            guidance,
            estimands,
            estimands_variance,
            spp,
        )
    elapsed = time.time() - start_time
    print(f"Filtering time: {elapsed:.4f} seconds")

    # Guardar resultado
    result = result.squeeze(0)
    result_np = result.permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)
    output_path = f"{base_io}/transient/{escena}/denoised_transient_{spp}_{spatial_radius}_{temporal_radius}_{alpha}.npy"
    np.save(output_path, result_np)
    print(f"Resultado guardado en {output_path}")
