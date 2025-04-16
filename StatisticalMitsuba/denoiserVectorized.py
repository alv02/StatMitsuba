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
    def __init__(self, in_planes, kernel_size=3):
        super(Shift, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        n, c, h, w = x.size()
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")

        cat_layers = []
        for y in range(self.kernel_size):
            y2 = y + h
            for x in range(self.kernel_size):
                x2 = x + w
                xx = x_pad[:, :, y:y2, x:x2]
                cat_layers.append(xx)

        return torch.cat(cat_layers, 1)


def calculate_critical_value(n_i, n_j, alpha=0.005):
    """Calculate critical value using t-distribution."""
    # Calculate degrees of freedom
    degrees_of_freedom = n_i + n_j - 2

    # Calculate critical value from t-distribution
    gamma_w = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    return torch.tensor(gamma_w, dtype=torch.float32)


def compute_w(
    estimand_i, estimand_j, estimand_i_variance, estimand_j_variance, epsilon=1e-8
):
    """Compute optimal weight w_ij between pixels i and j."""
    numerator = (
        2 * (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
    )
    denominator = 2 * (
        (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
    )

    # Avoid division by zero
    denominator = torch.clamp(denominator, min=epsilon)

    # Compute result
    result = numerator / denominator

    # Handle special case where both numerator and denominator are 0

    result = torch.where(
        (numerator == 0) & (denominator == 0),
        torch.full_like(numerator, 0.5),
        numerator / denominator,
    )

    return result


def compute_t_statistic(w_ij, epsilon=1e-8):
    """Calculate t-statistic for comparing pixels."""
    # Handle w_ij = 1 case (return infinity)
    infinity_mask = w_ij == 1

    # Calculate t-statistic
    t_stat = torch.sqrt((1 / (2 * (1 - w_ij + epsilon))) - 1)

    # Set infinity values
    t_stat = torch.where(infinity_mask, torch.tensor(float("inf")), t_stat)

    return t_stat


class JointBilateralFilterWithMembership(nn.Module):
    """
    Joint Bilateral Filter with Membership Functions uses guidance image to calculate weights
    and statistical tests to determine which pixels should be combined.
    """

    def __init__(self, radius=5, sigma_diag=None, alpha=0.005):
        super(JointBilateralFilterWithMembership, self).__init__()

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
        self.shift = Shift(
            8, self.kernel_size
        )  # 8 channels for guidance (pos, albedo, normals)

    def forward(self, image, guidance, estimands, estimands_variance, spp):
        """
        Args:
            image: Input image to denoise [B, C, H, W]
            guidance: Guidance features [B, G, H, W] where G is guidance channels
                     (positions, albedo, normals)
            estimands: Pre-computed estimands [B, C, H, W]
            estimands_variance: Pre-computed variance of estimands [B, C, H, W]
            spp: Samples per pixel (scalar)
        """
        # Calculate critical value for statistical test
        gamma_w = calculate_critical_value(spp, spp, self.alpha).to(image.device)

        # Get shifted guidance
        shifted_guidance = self.shift(guidance)
        n, c, h, w = guidance.size()
        n_patches = self.kernel_size * self.kernel_size

        # Reshape guidance to [B, G*n_patches, H, W] -> [B, n_patches, G, H, W]
        shifted_guidance = shifted_guidance.view(n, n_patches, c, h, w)

        # Get center pixel values from guidance
        center_idx = n_patches // 2
        center_guidance = shifted_guidance[:, center_idx : center_idx + 1, :, :, :]

        # Calculate pixel differences (p_j - p_i)
        diff = shifted_guidance - center_guidance

        # Reshape diff for matrix multiplication [B, n_patches, G, H, W] -> [B, n_patches, H, W, G]
        diff = diff.permute(0, 1, 3, 4, 2)
        batch, n_patches, height, width, guidance_channels = diff.shape
        diff_reshaped = diff.reshape(
            batch, n_patches, height * width, guidance_channels
        )

        # Calculate Mahalanobis distance: (p_j - p_i)^T * Σ^(-1) * (p_j - p_i)
        # [B, n_patches, H*W, G] x [G, G] -> [B, n_patches, H*W, G]
        temp = torch.matmul(diff_reshaped, self.sigma_inv)
        # Batch matrix multiply: [B, n_patches, H*W, G] x [B, n_patches, H*W, G] -> [B, n_patches, H*W]
        mahalanobis_sq = torch.sum(temp * diff_reshaped, dim=3)
        mahalanobis_sq = mahalanobis_sq.reshape(batch, n_patches, height, width)

        # Calculate bilateral filter weight: ρ_ij = exp(-0.5 * mahalanobis_sq)
        bilateral_weights = torch.exp(-0.5 * mahalanobis_sq)

        # Create shifted estimands and variances for statistical testing
        shifted_estimands = self.shift(estimands)
        shifted_estimands_variance = self.shift(estimands_variance)

        n, c_img, h, w = estimands.size()
        shifted_estimands = shifted_estimands.view(n, n_patches, c_img, h, w)
        shifted_estimands_variance = shifted_estimands_variance.view(
            n, n_patches, c_img, h, w
        )

        # Get center pixel estimands and variances
        center_estimands = estimands.unsqueeze(1)  # [B, 1, C, H, W]
        center_estimands_variance = estimands_variance.unsqueeze(1)  # [B, 1, C, H, W]

        # Calculate w_ij for each pixel pair and each channel
        # [B, n_patches, C, H, W], [B, 1, C, H, W] -> [B, n_patches, C, H, W]
        w_ij = compute_w(
            center_estimands,
            shifted_estimands,
            center_estimands_variance,
            shifted_estimands_variance,
        )

        # Calculate t-statistic
        t_stat = compute_t_statistic(w_ij)  # [B, n_patches, C, H, W]

        # Create membership function (m_ij)
        # Compare t-stat with gamma_w across all channels
        # t_stat: [B, n_patches, C, H, W], gamma_w: scalar
        membership = (
            (t_stat < gamma_w).all(dim=2, keepdim=True).float()
        )  # [B, n_patches, 1, H, W]

        # Set center pixel membership to 1
        membership[:, center_idx : center_idx + 1, :, :, :] = 1.0

        # Apply both bilateral weights and membership function
        # [B, n_patches, H, W] * [B, n_patches, 1, H, W] -> [B, n_patches, 1, H, W]
        bilateral_weights = bilateral_weights.unsqueeze(2) * membership

        # Apply weights to shifted input image
        shifted_image = self.shift(image)
        n, c_img, h, w = image.size()
        shifted_image = shifted_image.view(n, n_patches, c_img, h, w)

        # Apply combined weights to shifted image
        # [B, n_patches, C_img, H, W] * [B, n_patches, 1, H, W]
        weighted_values = shifted_image * bilateral_weights

        # Sum along patches dimension [B, n_patches, C_img, H, W] -> [B, C_img, H, W]
        weighted_sum = torch.sum(weighted_values, dim=1)

        # Sum of weights for normalization [B, n_patches, 1, H, W] -> [B, 1, H, W]
        sum_weights = torch.sum(bilateral_weights, dim=1)

        # Avoid division by zero
        sum_weights = torch.clamp(sum_weights, min=1e-10)

        # Calculate final denoised image [B, C_img, H, W] / [B, 1, H, W]
        denoised_image = weighted_sum / sum_weights

        return denoised_image


if __name__ == "__main__":
    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    # Load the EXR file
    bitmap = mi.Bitmap("./cbox.exr")

    # Load pre-computed statistics (already in channels-first format)
    statistics = np.load("./stats.npy")  # [C, H, W, 2]
    # Extract pre-computed statistics - just add batch dimension
    estimands = (
        torch.from_numpy(statistics[:, :, :, 0]).float().unsqueeze(0)
    )  # [1, C, H, W]
    estimands_variance = (
        torch.from_numpy(statistics[:, :, :, 1]).float().unsqueeze(0)
    )  # [1, C, H, W]

    spp = 32  # Replace with your actual spp value

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
    jbf_with_membership = JointBilateralFilterWithMembership(
        radius=5, sigma_diag=sigma_diag
    )

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
    result_bitmap.write("denoised_image.exr")
