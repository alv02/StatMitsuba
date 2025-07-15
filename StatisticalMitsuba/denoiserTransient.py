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
        self.gamma_w = self.compute_gamma_w(spp, spp, alpha)
        print(self.gamma_w)

        # Debug parameters

        # Sigma_inv(1, C*n_patches, 1, 1)
        sigma_inv = torch.tensor([1, 1, 1, 50, 50, 50, 10, 10, 10], dtype=torch.float32)
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
            # Fix: divide by tile size, not number of tiles
            tile_y = pixel_y // self.tile.final_spatial_tile_size
            tile_x = pixel_x // self.tile.final_spatial_tile_size
            tile_t = pixel_t // self.tile.final_temporal_tile_size

            # Calculate linear tile index
            index = dr.fma(tile_y, tiles_x, tile_x)
            index = dr.fma(index, tiles_t, tile_t)

            # Calculate offset within the tile
            offset_y = pixel_y % self.tile.final_spatial_tile_size
            offset_x = pixel_x % self.tile.final_spatial_tile_size
            offset_t = pixel_t % self.tile.final_temporal_tile_size

            tiled_debug_pixels.append(
                (index, offset_x, offset_y, offset_t, pixel_x, pixel_y, pixel_t)
            )

        return tiled_debug_pixels

    def debug(
        self,
        shifted_tile,
        shifted_estimands,
        shifted_estimands_variance,
        weights_jbf,
        membership,
        final_weights,
        current_tile_index,
        H,
        W,
        T,
    ):
        if debug_pixels is None:
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
            if tile_index != current_tile_index:
                continue

            def extract_patch(tensor, name):
                C, _, _ = tensor.shape
                tensor = tensor.reshape(
                    C,  # canales
                    -1,
                    self.tile.final_spatial_tile_size,
                    self.tile.final_spatial_tile_size,
                    self.tile.final_temporal_tile_size,
                )
                tensor = (
                    tensor[:, :, offset_y, offset_x, offset_t]
                    .reshape(
                        -1,
                        self.spatial_kernel_size,
                        self.spatial_kernel_size,
                        self.temporal_kernel_size,
                    )
                    .permute(1, 2, 3, 0)
                )

                np.save(
                    f"./debug_output/transient/{name}_{pixel_x}_{pixel_y}_{pixel_t}.npy",
                    tensor.cpu().numpy(),
                )

            extract_patch(shifted_tile, "patches")
            extract_patch(shifted_estimands, "estimands")
            extract_patch(shifted_estimands_variance, "estimands_variance")
            extract_patch(weights_jbf, "weights_jbf")
            extract_patch(membership, "membership")
            extract_patch(final_weights, "final_weights")

    def compute_gamma_w(self, n_i, n_j, alpha=0.005):
        """Calculate critical value using t-distribution."""
        # Calculate degrees of freedom
        degrees_of_freedom = n_i + n_j - 2

        # Calculate critical value from t-distribution
        gamma_w = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

        return torch.tensor(gamma_w, dtype=torch.float32)

    def compute_gamma(self, n_i, n_j, alpha=0.005):
        gamma_w = self.compute_gamma_w(n_i, n_j, alpha)
        return 1 / (2 * (gamma_w**2 + 1))

    def compute_w(
        self, estimand_i, estimand_j, estimand_i_variance, estimand_j_variance
    ):
        """Compute optimal weight w_ij between pixels i and j."""
        numerator = (
            2 * (estimand_i - estimand_j) ** 2
            + estimand_i_variance
            + estimand_j_variance
        )
        denominator = 2 * (
            (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
        )

        # Avoid division by zero
        denominator_safe = torch.where(denominator == 0, 1, denominator)

        # Compute result
        result = numerator / denominator_safe

        # Handle special case where both numerator and denominator are 0

        result = torch.where(
            (numerator == 0) & (denominator == 0.0),
            torch.full_like(numerator, 0.5),
            numerator / denominator,
        )

        variance_zero = (estimand_i_variance == 0.0) | (estimand_j_variance == 0.0)
        # variance_zero = (estimand_i_variance + estimand_j_variance) == 0.0
        values_differ = estimand_i != estimand_j
        result = torch.where(
            variance_zero & values_differ, torch.full_like(numerator, 1.0), result
        )

        return result

    def compute_t_statistic(self, w_ij):
        """Calculate t-statistic for comparing pixels."""
        inf_tensor = torch.full_like(w_ij, float("inf"))
        return torch.where(
            w_ij == 1.0,
            inf_tensor,  # Si w_ij es 1, devuelve infinito
            torch.sqrt((1 / (2 * (1 - w_ij))) - 1),
        )

    def compute_bilateral_weights(self, guidance):
        """Guidance (C, H, W, T)"""
        shifted_guidance = self.shift(guidance)

        # Get center guidance
        center_idx = self.n_patches // 2
        center_guidance = shifted_guidance[:, center_idx : center_idx + 1, :]

        diff = shifted_guidance - center_guidance

        diff_squared = diff**2

        # Apply weights (in-place operation)
        weighted_diff = diff_squared * self.sigma_inv

        result = weighted_diff.sum(dim=0)

        # Exponential (in-place)
        bilateral_weights = torch.exp_(
            -0.5 * result
        )  # Use exp_ for in-place if available

        return bilateral_weights

    def compute_membership(self, estimands, estimands_variance):
        center_idx = self.n_patches // 2

        shifted_estimands = self.shift(estimands)
        shifted_estimands_variance = self.shift(estimands_variance)

        center_estimands = shifted_estimands[:, center_idx : center_idx + 1, :]
        center_estimands_variance = shifted_estimands_variance[
            :, center_idx : center_idx + 1, :
        ]

        w_ij = self.compute_w(
            center_estimands,
            shifted_estimands,
            center_estimands_variance,
            shifted_estimands_variance,
        )
        t_stat = self.compute_t_statistic(w_ij)

        membership = (t_stat < self.gamma_w).all(dim=0, keepdim=True).float()
        membership[:, center_idx : center_idx + 1, :] = 1.0

        return membership

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
        tile_index = 0
        for y in range(tiles_y):
            for x in range(tiles_x):
                for t in range(tiles_t):

                    img_tile = tiled_images[y, x, t, :]
                    guidance_tile = tiled_guidance[y, x, t, :]
                    estimands_tile = tiled_estimands[y, x, t, :]
                    var_tile = tiled_estimands_variance[y, x, t, :]

                    # Calcular pesos
                    weights_jbf = self.compute_bilateral_weights(
                        guidance_tile
                    ).unsqueeze(0)
                    membership = self.compute_membership(estimands_tile, var_tile)
                    final_weights = weights_jbf * membership

                    # Obtener vecindario de píxeles de la imagen
                    shifted_image = self.shift(img_tile)

                    sum_weights = torch.sum(final_weights, dim=1)
                    final_weights = final_weights / sum_weights
                    weighted_values = shifted_image * final_weights
                    denoised_image = torch.sum(weighted_values, dim=1)
                    tiled_denoised_image[tile_index : tile_index + 1] = (
                        denoised_image.reshape(
                            C,
                            self.tile.final_spatial_tile_size,
                            self.tile.final_spatial_tile_size,
                            self.tile.final_temporal_tile_size,
                        )
                    )
                    self.debug(
                        shifted_image,
                        self.shift(estimands_tile),
                        self.shift(var_tile),
                        weights_jbf,
                        membership,
                        final_weights,
                        tile_index,
                        H,
                        W,
                        T,
                    )
                    tile_index += 1

        H_OUT = (
            math.ceil(H / self.tile.final_spatial_tile_size)
            * self.tile.final_spatial_tile_size
        )
        W_OUT = (
            math.ceil(W / self.tile.final_spatial_tile_size)
            * self.tile.final_spatial_tile_size
        )
        T_OUT = (
            math.ceil(T / self.tile.final_temporal_tile_size)
            * self.tile.final_temporal_tile_size
        )
        result = combine_tiles_3d(
            tiled_denoised_image,
            (
                self.tile.final_spatial_tile_size,
                self.tile.final_spatial_tile_size,
                self.tile.final_temporal_tile_size,
            ),
            (1, C, H_OUT, W_OUT, T_OUT),
            0,
            self.tile.stride,
        )
        result = result[:, :, 0:H, 0:W, 0:T]
        return result


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
    debug_pixels = [(238, 588, 0)]
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
