#!/usr/bin/python
# Joint Bilateral Filter with Membership Functions in PyTorch (Simplified)

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


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
        returns (B, C*K**2, H, W)
        """
        B, C, H, W = x.shape
        # Extract patches using unfold (similar to what you already had)
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.radius)
        patches = patches.view(1, C * self.n_patches, H, W)
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
        self.debug_pixels = debug_pixels  # List of (y, x) coordinates to debug

        # Sigma_inv(1, C*n_patches, 1, 1)
        self.sigma_inv = torch.tensor(
            [0.1, 0.1, 50, 50, 50, 10, 10, 10], dtype=torch.float32
        )
        self.sigma_inv = self.sigma_inv.repeat(self.n_patches)
        self.sigma_inv = self.sigma_inv.view(1, self.n_patches, -1, 1, 1)
        self.sigma_inv = torch.permute(self.sigma_inv, (0, 2, 1, 3, 4))
        self.sigma_inv = torch.reshape(self.sigma_inv, (1, -1, self.n_patches, 1, 1))

        # Create shift operator
        self.shift = Shift(radius)

    def debug(self, weights_jbf, membership, final_weights):
        _, _, _, H, W = membership.shape
        if self.debug_pixels:
            os.makedirs("debug_output", exist_ok=True)  # Crea la carpeta si no existe
            for x, y in self.debug_pixels:
                if 0 <= y < H and 0 <= x < W:
                    # Extracción de los pesos JBF
                    weights = (
                        weights_jbf[0, 0, :, y, x]
                        .cpu()
                        .numpy()
                        .reshape(self.kernel_size, self.kernel_size)
                    )
                    # Extracción de los pesos de membership
                    mem = (
                        membership[0, 0, :, y, x]
                        .cpu()
                        .numpy()
                        .reshape(self.kernel_size, self.kernel_size)
                    )
                    # Extracción de los pesos finales
                    final_w = (
                        final_weights[0, 0, :, y, x]
                        .cpu()
                        .numpy()
                        .reshape(self.kernel_size, self.kernel_size)
                    )

                    # Creación de la figura para mostrar los tres mapas
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f"Debug Pixel ({x}, {y})")

                    # Mostrar pesos JBF
                    im0 = axs[0].imshow(weights, cmap="viridis")
                    axs[0].set_title("Bilateral Weights")
                    fig.colorbar(im0, ax=axs[0])

                    # Mostrar los valores de membership
                    im1 = axs[1].imshow(mem, cmap="gray", vmin=0, vmax=1)
                    axs[1].set_title("Membership")
                    fig.colorbar(im1, ax=axs[1])

                    # Mostrar los pesos finales
                    im2 = axs[2].imshow(final_w, cmap="viridis")
                    axs[2].set_title("Final Weights")
                    fig.colorbar(im2, ax=axs[2])

                    # Guarda la figura en un archivo
                    fig_path = f"debug_output/pixel_{x}_{y}.png"
                    plt.savefig(fig_path, dpi=300)
                    plt.close(fig)  # Cierra la figura para liberar memoria

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
        n, c, h, w = guidance.size()
        shifted_guidance = shifted_guidance.view(n, c, self.n_patches, h, w)

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
        b, c, h, w = estimands.size()
        shifted_estimands = shifted_estimands.view(b, c, self.n_patches, h, w)
        shifted_estimands_variance = shifted_estimands_variance.view(
            b, c, self.n_patches, h, w
        )

        center_estimands = estimands.unsqueeze(2)
        center_estimands_variance = estimands_variance.unsqueeze(2)

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
        # Shapes (B, 1, n_patches, H, W)
        bilateral_weights = self.compute_bilateral_weights(guidance).unsqueeze(1)
        membership = self.compute_membership(estimands, estimands_variance, spp)
        final_weights = bilateral_weights * membership

        shifted_image = self.shift(image)
        n, c_img, h, w = image.size()
        shifted_image = shifted_image.view(n, c_img, self.n_patches, h, w)

        sum_weights = torch.sum(final_weights, dim=2)
        sum_weights = torch.clamp(sum_weights, min=1e-10)
        final_weights = final_weights / sum_weights
        weighted_values = shifted_image * final_weights
        denoised_image = torch.sum(weighted_values, dim=2)

        if self.debug_pixels:
            self.debug(bilateral_weights, membership, final_weights)
        return denoised_image


if __name__ == "__main__":
    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    scene = "./volumetric"

    # Load the EXR file
    bitmap = mi.Bitmap(scene + ".exr")

    # Load pre-computed statistics (already in channels-first format)
    statistics = np.load(scene + "_stats.npy")  # [C, H, W, 3]
    estimands = (
        torch.from_numpy(statistics[:, :, :, 0]).to(torch.float32).unsqueeze(0)
    )  # [1, C, H, W]
    estimands_variance = (
        torch.from_numpy(statistics[:, :, :, 1]).to(torch.float32).unsqueeze(0)
    )  # [1, C, H, W]
    spp = statistics[0, 0, 0, 2]

    # Extract channels from EXR
    res = dict(bitmap.split())

    # Convert to PyTorch tensors
    image = (
        torch.from_numpy(np.array(res["<root>"], dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
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
    y_coords = torch.linspace(0, h - 1, h, dtype=torch.float32)
    x_coords = torch.linspace(0, w - 1, w, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    pos = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)

    # Concatenate guidance features
    guidance = torch.cat([pos, albedo, normals], dim=1)  # [1, 8, H, W]

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define debug pixels - modify these to the coordinates you want to examine
    debug_pixels = [(200, 378), (428, 1160)]

    # Initialize joint bilateral filter with membership
    stat_denoiser = StatDenoiser(radius=5, debug_pixels=debug_pixels)

    # Move tensors and model to device
    stat_denoiser = stat_denoiser.to(device)
    image = image.to(device)
    guidance = guidance.to(device)
    estimands = estimands.to(device)
    estimands_variance = estimands_variance.to(device)

    # Time the filtering operation
    start_time = time.time()
    with torch.no_grad():
        result = stat_denoiser(image, guidance, estimands, estimands_variance, spp)
    elapsed = time.time() - start_time
    print(f"Filtering time: {elapsed:.4f} seconds")

    # Convert result back to CPU and numpy
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # Save result as EXR
    result_bitmap = mi.Bitmap(result_np)
    result_bitmap.write("denoised_image_v.exr")
