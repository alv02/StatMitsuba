# Joint Bilateral Filter with Membership Functions in PyTorch
import time

import matplotlib

matplotlib.use("Agg")
import math

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


class Tile(nn.Module):
    """
    Creates a tiled tensor for and image
    """

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.final_tile_size = 128
        self.tile_size = self.final_tile_size + 2 * radius
        self.stride = self.final_tile_size

    def forward(self, x):
        """
        Returns tensor (tiles, C, H, W)
        """
        B, C, H, W = x.shape
        x_padded = F.pad(
            x,
            (
                self.radius,
                self.final_tile_size - 1 + self.radius,
                self.radius,
                self.final_tile_size - 1 + self.radius,
            ),
        )

        tiles = F.unfold(x_padded, kernel_size=(self.tile_size), stride=self.stride)
        tiles = tiles.view(1, C, self.tile_size, self.tile_size, -1)
        tiles = tiles.permute(0, 4, 1, 2, 3)
        tiles = tiles.reshape(-1, C, self.tile_size, self.tile_size)
        return tiles


class Shift(nn.Module):
    """
    Creates a tensor with the neighbours of each pixel
    """

    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.n_patches = self.kernel_size**2

    def forward(self, x):
        """
        x (B, C, H, W)
        returns (B, C*n_patches, H, W)
        """
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size)
        patches = patches.view(
            1, C * self.n_patches, H - self.radius * 2, W - self.radius * 2
        )
        return patches


class StatDenoiser(nn.Module):
    """
    Joint Bilateral Filter with Membership Functions uses guidance image to calculate weights
    and statistical tests to determine which pixels should be combined.
    """

    def __init__(self, radius=5, alpha=0.005, debug_pixels=None):
        super(StatDenoiser, self).__init__()

        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.alpha = alpha
        self.n_patches = self.kernel_size**2

        # Debug parameters

        # Sigma_inv(1, C*n_patches, 1, 1)
        sigma_inv = torch.tensor(
            [0.1, 0.1, 50, 50, 50, 10, 10, 10], dtype=torch.float32
        )
        sigma_inv = sigma_inv.repeat(self.n_patches)
        sigma_inv = sigma_inv.view(1, self.n_patches, -1, 1, 1)
        sigma_inv = torch.permute(sigma_inv, (0, 2, 1, 3, 4))
        sigma_inv = torch.reshape(sigma_inv, (1, -1, self.n_patches, 1, 1))

        self.register_buffer("sigma_inv", sigma_inv)
        # Create shift operator
        self.shift = Shift(radius)
        self.tile = Tile(radius)
        self.debug_pixels = debug_pixels

    def get_debug_pixels_tiled(self, debug_pixels, W):
        tiled_debug_pixels = []
        tiles_per_row = math.ceil(W / self.tile.final_tile_size)

        for pixel_x, pixel_y in debug_pixels:
            tile_x = pixel_x // self.tile.final_tile_size
            tile_y = pixel_y // self.tile.final_tile_size
            tile_linear_index = tile_y * tiles_per_row + tile_x

            offset_x = pixel_x % self.tile.final_tile_size
            offset_y = pixel_y % self.tile.final_tile_size

            # También devolvemos la posición absoluta
            tiled_debug_pixels.append(
                (tile_linear_index, offset_x, offset_y, pixel_x, pixel_y)
            )

        return tiled_debug_pixels

    def debug(self, weights_jbf, membership, final_weights, tile_index, W):
        if self.debug_pixels is None:
            return

        debug_pixels_tiled = self.get_debug_pixels_tiled(self.debug_pixels, W)

        for pixel_tile, offset_x, offset_y, abs_x, abs_y in debug_pixels_tiled:
            if pixel_tile == tile_index:
                weights = (
                    weights_jbf[0, 0, :, offset_y, offset_x]
                    .cpu()
                    .numpy()
                    .reshape(self.kernel_size, self.kernel_size)
                )
                mem = (
                    membership[0, 0, :, offset_y, offset_x]
                    .cpu()
                    .numpy()
                    .reshape(self.kernel_size, self.kernel_size)
                )
                final_w = (
                    final_weights[0, 0, :, offset_y, offset_x]
                    .cpu()
                    .numpy()
                    .reshape(self.kernel_size, self.kernel_size)
                )

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Debug Pixel (abs_x={abs_x}, abs_y={abs_y})")

                im0 = axs[0].imshow(weights, cmap="viridis")
                axs[0].set_title("Bilateral Weights")
                fig.colorbar(im0, ax=axs[0])

                im1 = axs[1].imshow(mem, cmap="gray", vmin=0, vmax=1)
                axs[1].set_title("Membership")
                fig.colorbar(im1, ax=axs[1])

                im2 = axs[2].imshow(final_w, cmap="viridis")
                axs[2].set_title("Final Weights")
                fig.colorbar(im2, ax=axs[2])

                # Guardar usando coordenadas absolutas
                fig_path = f"debug_output/pixel_{abs_x}_{abs_y}.png"
                plt.savefig(fig_path, dpi=300)
                plt.close(fig)

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
            (numerator == 0) & (denominator == 0),
            torch.full_like(numerator, 0.5),
            numerator / denominator,
        )

        variance_zero = (estimand_i_variance == 0) | (estimand_j_variance == 0)
        values_differ = estimand_i != estimand_j
        result = torch.where(
            variance_zero & values_differ, torch.full_like(numerator, 1), result
        )

        return result

    def compute_t_statistic(self, w_ij):
        """Calculate t-statistic for comparing pixels."""
        return torch.where(
            w_ij == 1,
            float("inf"),  # Si w_ij es 1, devuelve infinito
            torch.sqrt((1 / (2 * (1 - w_ij))) - 1),
        )

    def compute_bilateral_weights(self, guidance):
        shifted_guidance = self.shift(guidance)
        n, c, _, _ = guidance.size()
        shifted_guidance = shifted_guidance.view(
            n, c, self.n_patches, self.tile.final_tile_size, self.tile.final_tile_size
        )

        # Get center guidance
        center_idx = self.n_patches // 2
        center_guidance = shifted_guidance[:, :, center_idx : center_idx + 1, :, :]

        diff = shifted_guidance - center_guidance

        diff_squared = diff**2

        # Apply weights (in-place operation)
        weighted_diff = diff_squared * self.sigma_inv

        result = weighted_diff.sum(dim=1)

        # Exponential (in-place)
        bilateral_weights = torch.exp_(
            -0.5 * result
        )  # Use exp_ for in-place if available

        return bilateral_weights

    def compute_membership(self, estimands, estimands_variance, spp):
        gamma_w = self.compute_gamma_w(spp, spp, self.alpha).to(estimands.device)
        center_idx = self.n_patches // 2

        shifted_estimands = self.shift(estimands)
        shifted_estimands_variance = self.shift(estimands_variance)
        b, _, h, w = shifted_estimands.shape
        shifted_estimands = shifted_estimands.view(b, -1, self.n_patches, h, w)
        shifted_estimands_variance = shifted_estimands_variance.view(
            b, -1, self.n_patches, h, w
        )

        center_estimands = estimands[
            :, :, self.radius : -self.radius, self.radius : -self.radius
        ].unsqueeze(2)
        center_estimands_variance = estimands_variance[
            :, :, self.radius : -self.radius, self.radius : -self.radius
        ].unsqueeze(2)

        w_ij = self.compute_w(
            center_estimands,
            shifted_estimands,
            center_estimands_variance,
            shifted_estimands_variance,
        )
        t_stat = self.compute_t_statistic(w_ij)

        membership = (t_stat < gamma_w).all(dim=1, keepdim=True).float()
        membership[:, :, center_idx : center_idx + 1, :, :] = 1.0

        return membership

    def forward(self, image, guidance, estimands, estimands_variance, spp):
        # Tile inputs
        tiled_image = self.tile(image)
        tiled_guidance = self.tile(guidance)
        tiled_estimands = self.tile(estimands)
        tiled_estimands_variance = self.tile(estimands_variance)
        _, _, H, W = image.shape

        # Procesamiento por tile
        batch_tiles, C, _, _ = tiled_image.shape
        tiled_denoised_image = torch.empty(
            (batch_tiles, C, self.tile.final_tile_size, self.tile.final_tile_size)
        )

        for i in range(batch_tiles):
            img_tile = tiled_image[i : i + 1]
            guidance_tile = tiled_guidance[i : i + 1]
            estimands_tile = tiled_estimands[i : i + 1]
            var_tile = tiled_estimands_variance[i : i + 1]

            # Calcular pesos
            weights_jbf = self.compute_bilateral_weights(guidance_tile).unsqueeze(1)
            membership = self.compute_membership(estimands_tile, var_tile, spp)
            final_weights = weights_jbf * membership

            # Obtener vecindario de píxeles de la imagen
            shifted_image = self.shift(img_tile)
            b, c, h, w = img_tile.shape
            shifted_image = shifted_image.view(
                b, c, self.n_patches, h - 2 * self.radius, w - 2 * self.radius
            )

            sum_weights = torch.sum(final_weights, dim=2)
            sum_weights = torch.clamp(sum_weights, min=1e-10)
            final_weights = final_weights / sum_weights
            weighted_values = shifted_image * final_weights
            denoised_image = torch.sum(weighted_values, dim=2)
            tiled_denoised_image[i : i + 1] = denoised_image[0]

            self.debug(weights_jbf, membership, final_weights, i, W)

        # Tiled denoised image (tiles, C, height, width) fold necesita (b, C*height_width, tiles)
        tiled_denoised_image = (
            tiled_denoised_image.permute(1, 2, 3, 0)
            .reshape((C * self.tile.final_tile_size**2, batch_tiles))
            .unsqueeze(0)
        )

        H_OUT = math.ceil(H / self.tile.final_tile_size) * self.tile.final_tile_size
        W_OUT = math.ceil(W / self.tile.final_tile_size) * self.tile.final_tile_size
        denoised_image = F.fold(
            tiled_denoised_image,
            (H_OUT, W_OUT),
            kernel_size=self.tile.final_tile_size,
            stride=self.tile.final_tile_size,
        )

        return denoised_image


if __name__ == "__main__":
    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    scene = "./aovs-transient"

    # Load the EXR file
    bitmap = mi.Bitmap(scene + ".exr")

    # Load pre-computed statistics (already in channels-first format)
    statistics = np.load("./transient_stats.npy")  # [H, W, T,C, 3]
    estimands = (
        torch.from_numpy(statistics[..., 0]).to(torch.float32).permute(2, 3, 0, 1)
    )
    estimands_variance = (
        torch.from_numpy(statistics[..., 1]).to(torch.float32).permute(2, 3, 0, 1)
    )  # [1, C, H, W]
    spp = statistics[0, 0, 0, 0, 2]
    # Extract channels from EXR
    res = dict(bitmap.split())

    # Convert to PyTorch tensors
    images = np.load("./transient_data.npy")
    images = torch.from_numpy(images).to(torch.float32).permute(2, 3, 0, 1)

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

    # Generate position features
    h, w = albedo.shape[2], albedo.shape[3]

    y_coords = torch.linspace(-1, 1, h, dtype=torch.float32)
    x_coords = torch.linspace(-w / h, w / h, w, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    pos = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    # Concatenate guidance features
    guidance = torch.cat([pos, albedo, normals], dim=1)  # [1, 8, H, W]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define debug pixels - modify these to the coordinates you want to examine
    debug_pixels = [(57, 344)]

    # Initialize joint bilateral filter with membership
    stat_denoiser = StatDenoiser(radius=20, alpha=0.005, debug_pixels=debug_pixels)

    # Move tensors and model to device

    final_result = torch.zeros_like(images)
    batches, _, _, _ = images.shape
    start_time = time.time()
    i = 50
    image_per_batch = images[i, ...].unsqueeze(0)
    estimands_per_batch = estimands[i, ...].unsqueeze(0)
    estimands_variance_per_batch = estimands_variance[i, ...].unsqueeze(0)
    stat_denoiser = stat_denoiser.to(device)
    image_per_batch = image_per_batch.to(device)
    guidance = guidance.to(device)
    estimands_per_batch = estimands_per_batch.to(device)
    estimands_variance_per_batch = estimands_variance_per_batch.to(device)

    # Time the filtering operation
    with torch.no_grad():
        result = stat_denoiser(
            image_per_batch,
            guidance,
            estimands_per_batch,
            estimands_variance_per_batch,
            spp,
        )
    final_result[i, ...] = result[..., 0:h, 0:w]

    elapsed = time.time() - start_time
    print(f"Filtering time: {elapsed:.4f} seconds")

    final_result_np = final_result.permute(2, 3, 0, 1).cpu().numpy().astype(np.float32)
    np.save("denoised_transient.npy", final_result_np)

    result_bitmap = mi.Bitmap(final_result_np[:, :, i, ...])
    original_np = (
        images[i, ...].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    )
    original_bitmap = mi.Bitmap(original_np)
    result_bitmap.write("denoised_transient_image.exr")
    original_bitmap.write("original_transient.exr")
