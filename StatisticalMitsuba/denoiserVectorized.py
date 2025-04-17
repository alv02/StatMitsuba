#!/usr/bin/python
# Joint Bilateral Filter with Membership Functions in PyTorch (Simplified)

import time

import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


class Shift(nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.kernel_size = 2 * radius + 1

    def forward(self, x):
        B, C, H, W = x.shape
        # Use reflection padding like in the original
        x_pad = F.pad(
            x, (self.radius, self.radius, self.radius, self.radius), mode="reflect"
        )

        # Extract patches using unfold (similar to what you already had)
        patches = F.unfold(x_pad, kernel_size=self.kernel_size, padding=0)

        # Reshape to [B, C, K*K, H, W] with proper ordering
        patches = patches.view(B, C, self.kernel_size**2, H, W)

        # Reshape to match your original expected output format
        patches = patches.permute(0, 2, 1, 3, 4).reshape(
            B, self.kernel_size**2 * C, H, W
        )

        return patches


class StatDenoiser(nn.Module):
    """
    Joint Bilateral Filter with Membership Functions uses guidance image to calculate weights
    and statistical tests to determine which pixels should be combined.
    """

    def __init__(self, radius=5, sigma_diag=None, alpha=0.005):
        super(StatDenoiser, self).__init__()

        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.alpha = alpha

        # Default sigma values if not provided
        if sigma_diag is None:
            sigma_diag = torch.tensor(
                [10.0, 10.0, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1], dtype=torch.float32
            )

        # Create sigma matrix and its inverse
        sigma = torch.diag(sigma_diag)
        self.register_buffer("sigma_inv", torch.inverse(sigma))

        # Create shift operator
        self.shift = Shift(radius)

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
        n_patches = self.kernel_size * self.kernel_size
        shifted_guidance = shifted_guidance.view(n, n_patches, c, h, w)

        center_idx = n_patches // 2
        center_guidance = shifted_guidance[:, center_idx : center_idx + 1, :, :, :]
        diff = shifted_guidance - center_guidance
        diff = diff.permute(0, 1, 3, 4, 2)
        batch, n_patches, height, width, guidance_channels = diff.shape
        diff_reshaped = diff.reshape(
            batch, n_patches, height * width, guidance_channels
        )

        temp = torch.matmul(diff_reshaped, self.sigma_inv)
        mahalanobis_sq = torch.sum(temp * diff_reshaped, dim=3)
        mahalanobis_sq = mahalanobis_sq.reshape(batch, n_patches, height, width)

        return torch.exp(-0.5 * mahalanobis_sq)  # bilateral_weights

    def compute_membership(self, estimands, estimands_variance, spp):
        gamma_w = self.compute_gamma_w(spp, spp, self.alpha).to(estimands.device)
        n_patches = self.kernel_size * self.kernel_size
        center_idx = n_patches // 2

        shifted_estimands = self.shift(estimands)
        shifted_estimands_variance = self.shift(estimands_variance)
        n, c, h, w = estimands.size()
        shifted_estimands = shifted_estimands.view(n, n_patches, c, h, w)
        shifted_estimands_variance = shifted_estimands_variance.view(
            n, n_patches, c, h, w
        )

        center_estimands = estimands.unsqueeze(1)
        center_estimands_variance = estimands_variance.unsqueeze(1)

        w_ij = self.compute_w(
            center_estimands,
            shifted_estimands,
            center_estimands_variance,
            shifted_estimands_variance,
        )
        t_stat = self.compute_t_statistic(w_ij)

        membership = (t_stat < gamma_w).all(dim=2, keepdim=True).float()
        membership[:, center_idx : center_idx + 1, :, :, :] = 1.0

        return membership

    def forward(self, image, guidance, estimands, estimands_variance, spp):
        n_patches = self.kernel_size * self.kernel_size
        bilateral_weights = self.compute_bilateral_weights(guidance).unsqueeze(2)
        membership = self.compute_membership(estimands, estimands_variance, spp)
        bilateral_weights = bilateral_weights * membership

        shifted_image = self.shift(image)
        n, c_img, h, w = image.size()
        shifted_image = shifted_image.view(n, n_patches, c_img, h, w)

        weighted_values = shifted_image * bilateral_weights
        weighted_sum = torch.sum(weighted_values, dim=1)
        sum_weights = torch.sum(bilateral_weights, dim=1)
        sum_weights = torch.clamp(sum_weights, min=1e-10)

        denoised_image = weighted_sum / sum_weights
        return denoised_image


if __name__ == "__main__":
    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    # Load the EXR file
    bitmap = mi.Bitmap("./staircase.exr")

    # Load pre-computed statistics (already in channels-first format)
    statistics = np.load("./stats_staircase.npy")  # [C, H, W, 3]
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

    # Initialize joint bilateral filter with membership
    sigma_diag = torch.tensor(
        [10.0, 10.0, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1], dtype=torch.float32
    )
    jbf_with_membership = StatDenoiser(radius=20, sigma_diag=sigma_diag)

    # Move tensors and model to device
    jbf_with_membership = jbf_with_membership.to(device)
    image = image.to(device)
    guidance = guidance.to(device)
    estimands = estimands.to(device)
    estimands_variance = estimands_variance.to(device)

    # Time the filtering operation
    start_time = time.time()
    with torch.no_grad():
        result = jbf_with_membership(
            image, guidance, estimands, estimands_variance, spp
        )
    elapsed = time.time() - start_time
    print(f"Filtering time: {elapsed:.4f} seconds")

    # Convert result back to CPU and numpy
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # Save result as EXR
    result_bitmap = mi.Bitmap(result_np)
    result_bitmap.write("denoised_image_v.exr")
