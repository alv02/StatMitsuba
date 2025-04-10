import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F


def apply_weights_to_image(image, weights, radius=5):
    """
    Aplica las weights a cada píxel en la imagen usando convolución.

    Parameters:
    - image: La imagen original en formato numpy array (altura, ancho, canales)
    - weights: Matriz de pesos (altura, ancho, ventana, ventana)

    Returns:
    - La nueva imagen después de aplicar los pesos.
    """
    # Asumiendo que 'image' es una imagen numpy (height, width, channels)
    height, width, channels = image.shape

    # Los pesos tienen una forma de (height, width, radius * 2 + 1, radius * 2 + 1)
    # Que corresponde a una vecindad de cada píxel
    result_image = np.zeros_like(image)

    # Iteramos sobre cada píxel de la imagen
    for i in range(height):
        for j in range(width):
            # Extraemos la ventana de pesos para el píxel (i, j)
            weight_window = weights[i, j]

            # Extraemos la vecindad de píxeles correspondientes en la imagen original
            # En este caso, creamos una ventana centrada en (i, j) con el mismo tamaño que los pesos
            # Asumimos que la imagen tiene un rango de 3 canales (RGB)

            # Creamos una máscara para los píxeles válidos
            start_i = max(i - radius, 0)
            end_i = min(i + radius + 1, height)
            start_j = max(j - radius, 0)
            end_j = min(j + radius + 1, width)

            # Definir los límites dentro de la ventana de pesos
            start_i_w = max(radius - i, 0)
            end_i_w = start_i_w + (end_i - start_i)
            start_j_w = max(radius - j, 0)
            end_j_w = start_j_w + (end_j - start_j)
            # Extraemos la vecindad de la imagen
            neighborhood = image[start_i:end_i, start_j:end_j]

            weight_window_sub = weight_window[start_i_w:end_i_w, start_j_w:end_j_w]

            weight_window_expanded = np.expand_dims(weight_window_sub, axis=-1)

            # Ahora se puede realizar la multiplicación entre neighborhood y weight_window_expanded
            weighted_sum = np.sum(neighborhood * weight_window_expanded, axis=(0, 1))

            # Guardamos el resultado en el nuevo píxel
            result_image[i, j] = weighted_sum

    return result_image


def evaluate_base_filter(prior_i, prior_j, sigma_inv):
    # Compute the difference (p_j - p_i)
    diff = prior_j - prior_i

    # Calculate the weight using the Gaussian falloff formula
    weight = np.exp(-0.5 * (diff.T @ sigma_inv @ diff))

    return weight


def denoiser(albedo, normals, radius=1):
    print("Starting denoiser")
    device = torch.device("cpu")
    h, w, _ = albedo.shape
    ksize = 2 * radius + 1
    patch_size = ksize * ksize

    y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    albedo = np.ascontiguousarray(albedo)
    normals = np.ascontiguousarray(normals)

    print("Before feature stack")
    features = (
        torch.stack(
            [
                x_coords,
                y_coords,
                torch.from_numpy(albedo[..., 0]),
                torch.from_numpy(albedo[..., 1]),
                torch.from_numpy(albedo[..., 2]),
                torch.from_numpy(normals[..., 0]),
                torch.from_numpy(normals[..., 1]),
                torch.from_numpy(normals[..., 2]),
            ],
            dim=-1,
        )
        .float()
        .to(device)
    )
    print("After feature stack")

    features = features.permute(2, 0, 1).unsqueeze(0)  # (1, 8, H, W)
    unfolded = F.unfold(
        features, kernel_size=ksize, padding=radius
    )  # (1, 8*ksize*ksize, H*W)
    unfolded = unfolded.view(1, 8, patch_size, h * w)

    print("Before centers")
    centers = features.view(1, 8, 1, h * w)
    print("After centers")

    diffs = unfolded - centers.expand_as(
        unfolded
    )  # Aseguramos que 'centers' se expanda para coincidir con 'unfolded'
    print("After diffs")

    sigma = torch.diag(torch.tensor([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])).to(
        device
    )
    sigma_inv = torch.linalg.inv(sigma)
    print("After sigma_inv")

    # Verificar las dimensiones de diffs y sigma_inv antes de la operación de multiplicación
    print(f"diffs shape: {diffs.shape}")
    print(f"sigma_inv shape: {sigma_inv.shape}")

    # Vamos a hacer la multiplicación de manera explícita
    diffs_reshaped = diffs.permute(3, 2, 1)  # (H*W, patch, 8)
    sigma_expanded = sigma_inv.unsqueeze(0).expand(h * w, -1, -1)  # (H*W, 8, 8)

    print(f"diffs_reshaped shape: {diffs_reshaped.shape}")
    print(f"sigma_expanded shape: {sigma_expanded.shape}")

    # Multiplicación explícita de matrices
    diff_sigma = torch.bmm(
        diffs_reshaped, sigma_expanded
    )  # (H*W, patch, 8) @ (H*W, 8, 8)
    mahalanobis = torch.bmm(
        diff_sigma, diffs_reshaped.permute(0, 2, 1)
    )  # (H*W, patch, 8) @ (H*W, 8, patch)
    print("After matrix multiplication")

    mahalanobis = mahalanobis.sum(dim=2)  # Sumamos a lo largo de la dimensión del patch
    mahalanobis = mahalanobis.view(h * w, patch_size)  # (H*W, patch_size)

    return mahalanobis


if __name__ == "__main__":
    dummy = np.ones((10, 10, 3), dtype=np.float32)
    dummy_normals = np.zeros((10, 10, 3), dtype=np.float32)
    dummy_normals[..., 2] = 1.0

    print("BEFORE DENOISER")
    result = denoiser(dummy, dummy_normals, radius=1)
    print("AFTER DENOISER")
