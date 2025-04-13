#!/usr/bin/python
# Joint Bilateral Filter in PyTorch

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


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


class JointBilateralFilter(nn.Module):
    """
    Joint Bilateral Filter uses guidance image to calculate weights
    and applies them to the input image.
    """

    def __init__(self, radius=5, sigma_diag=None):
        super(JointBilateralFilter, self).__init__()

        self.radius = radius
        self.kernel_size = 2 * radius + 1

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

    def forward(self, image, guidance):
        """
        Args:
            image: Input image to denoise [B, C, H, W]
            guidance: Guidance features [B, G, H, W] where G is guidance channels
                      (positions, albedo, normals)
        """
        # Get shifted guidance
        shifted_guidance = self.shift(guidance)
        n, c, h, w = guidance.size()
        n_patches = self.kernel_size * self.kernel_size

        # Reshape guidance to [B, G, H, W] -> [B, n_patches, G, H, W]
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

        # Calculate weight: ρ_ij = exp(-0.5 * mahalanobis_sq)
        weights = torch.exp(-0.5 * mahalanobis_sq)

        # Apply weights to shifted input image
        shifted_image = self.shift(image)
        n, c_img, h, w = image.size()
        shifted_image = shifted_image.view(n, n_patches, c_img, h, w)

        # Expand weights to match image channels [B, n_patches, H, W] -> [B, n_patches, 1, H, W]
        weights_expanded = weights.unsqueeze(2)

        # Apply weights to shifted image [B, n_patches, C_img, H, W] * [B, n_patches, 1, H, W]
        weighted_values = shifted_image * weights_expanded

        # Sum along patches dimension [B, n_patches, C_img, H, W] -> [B, C_img, H, W]
        weighted_sum = torch.sum(weighted_values, dim=1)

        # Sum of weights for normalization [B, n_patches, H, W] -> [B, 1, H, W]
        sum_weights = torch.sum(weights, dim=1, keepdim=True)

        # Avoid division by zero
        sum_weights = torch.clamp(sum_weights, min=1e-10)

        # Calculate final denoised image [B, C_img, H, W] / [B, 1, H, W]
        denoised_image = weighted_sum / sum_weights

        return denoised_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mitsuba as mi
    import numpy as np

    # Set Mitsuba variant
    mi.set_variant("llvm_ad_rgb")

    # Load the EXR file
    bitmap = mi.Bitmap("./staircase.exr")
    res = dict(bitmap.split())

    # Extract channels and convert to PyTorch tensors
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
    image = (
        torch.from_numpy(np.array(res["<root>"], dtype=np.float32))
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

    # Initialize joint bilateral filter
    sigma_diag = torch.tensor(
        [10.0, 10.0, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1], dtype=torch.float32
    )
    jbf = JointBilateralFilter(radius=20, sigma_diag=sigma_diag)

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move tensors and model to device
    jbf = jbf.to(device)
    image = image.to(device)
    guidance = guidance.to(device)

    # Time the filtering operation
    start_time = time.time()
    with torch.no_grad():
        result = jbf(image, guidance)
    elapsed = time.time() - start_time
    print(f"Filtering time: {elapsed:.4f} seconds")

    # Convert result back to CPU and numpy
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # Save result as EXR
    result_bitmap = mi.Bitmap(result_np)
    result_bitmap.write("denoised_pytorch.exr")

    # Optional: display result
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.array(res["<root>"]))
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(result_np)
    plt.title("Denoised Image")
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()
