import os

import mitsuba as mi
import numpy as np
import scipy.stats as stats
from numba import njit


@njit
def box_cox(samples, lam=0.5):
    if lam == 0.0:
        return np.log(samples).astype(np.float32)
    else:
        return ((np.power(samples, lam) - 1) / lam).astype(np.float32)


@njit
def calculate_statistics(samples, lam=0.5):
    """Calculate the mean, variance (M2), and skewness (M3) for each pixel based on MC samples."""
    channels, height, width, spp = samples.shape

    # Apply Box-Cox transformation to all samples at once (vectorized)
    samples_bc = box_cox(samples, lam)

    # Calcular la media manualmente
    mu = np.empty((channels, height, width))
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                s = 0.0
                for k in range(spp):
                    s += samples_bc[c, i, j, k]
                mu[c, i, j] = s / spp

    # Calcular M2 (varianza) y M3 (sesgo)
    M2 = np.empty((channels, height, width))
    M3 = np.empty((channels, height, width))
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                s2 = 0.0
                s3 = 0.0
                m = mu[c, i, j]
                for k in range(spp):
                    delta = samples_bc[c, i, j, k] - m
                    s2 += delta**2
                    s3 += delta**3
                M2[c, i, j] = s2 / spp
                M3[c, i, j] = s3 / spp

    # Varianza con corrección de Bessel (n-1)
    variance = np.empty((channels, height, width))
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                s = 0.0
                m = mu[c, i, j]
                for k in range(spp):
                    delta = samples_bc[c, i, j, k] - m
                    s += delta**2
                variance[c, i, j] = s / (spp - 1)

    # Transponer a (height, width, channels)
    mu = np.transpose(mu, (1, 2, 0))
    variance = np.transpose(variance, (1, 2, 0))
    M2 = np.transpose(M2, (1, 2, 0))
    M3 = np.transpose(M3, (1, 2, 0))

    return spp, mu, variance, M2, M3


@njit
def evaluate_base_filter(prior_i, prior_j, sigma_inv):
    # Compute the difference (p_j - p_i)
    diff = prior_j - prior_i

    # Calculate the weight using the Gaussian falloff formula
    weight = np.exp(-0.5 * (diff.T @ sigma_inv @ diff))

    return weight


def calculate_critical_value(n_i, n_j, alpha=0.005):
    # Calcula los grados de libertad
    degrees_of_freedom = n_i + n_j - 2

    # Calcula el valor crítico de la distribución t
    gamma_w = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    return gamma_w


@njit
def compute_w(estimand_i, estimand_j, estimand_i_variance, estimand_j_variance):
    numerator = (
        2 * (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
    )
    denominator = 2 * (
        (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
    )

    # Evitar división por 0 pero mantener los casos donde ambos sean 0
    denominator_safe = np.where(denominator == 0, 1, denominator)

    # Computar resultado normal
    result = numerator / denominator_safe

    # Si en un canal específico el numerador y el denominador son 0, ese canal se pone en 1
    result = np.where((numerator == 0) & (denominator == 0), 1, result)

    return result


@njit
def compute_t_statistic(w_ij):
    """Calcula el estadístico t para comparar píxeles en el filtro."""
    return np.where(
        w_ij == 1,
        float("inf"),  # Si w_ij es 1, devuelve infinito
        np.sqrt((1 / (2 * (1 - w_ij))) - 1),
    )


@njit
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

            for c in range(channels):
                acc = 0.0
                for x in range(neighborhood.shape[0]):
                    for y in range(neighborhood.shape[1]):
                        acc += neighborhood[x, y, c] * weight_window_sub[x, y]
                result_image[i, j, c] = acc

    return result_image


@njit
def denoiser(albedo, normals, n, mu, variance, M2, M3, gamma_w, sigma, radius=5):
    height, width, _ = np.shape(albedo)
    kernels = np.zeros((height, width, radius * 2 + 1, radius * 2 + 1))
    memebership_funcion = np.zeros((height, width, radius * 2 + 1, radius * 2 + 1))

    epsilon = 1e-8  # Pasar a parametro
    variance = np.where(variance == 0, epsilon, variance)
    sigma_inv = np.linalg.inv(sigma)

    # Pyxel i
    for i in range(height):
        for j in range(width):
            prior_i = np.array(
                [
                    j,
                    i,  # coordenadas (x, y)
                    albedo[i, j, 0],
                    albedo[i, j, 1],
                    albedo[i, j, 2],  # r, g, b
                    normals[i, j, 0],
                    normals[i, j, 1],
                    normals[i, j, 2],  # nx, ny, nz
                ]
            )
            # Neighbour pyxels j
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    if ni == i and nj == j:
                        kernels[i, j, di + radius, dj + radius] = 1
                        memebership_funcion[i, j, di + radius, dj + radius] = 1
                        continue

                    if not (0 <= ni < height and 0 <= nj < width):
                        continue
                    # Prior information for the neighboring pixel (i + di, j + dj)
                    prior_j = np.array(
                        [
                            nj,
                            ni,  # coordinates (x, y) - switched i, j
                            albedo[ni, nj, 0],
                            albedo[ni, nj, 1],
                            albedo[ni, nj, 2],  # r, g, b
                            normals[ni, nj, 0],
                            normals[ni, nj, 1],
                            normals[ni, nj, 2],  # nx, ny, nz
                        ]
                    )

                    kernels[i, j, di + radius, dj + radius] = evaluate_base_filter(
                        prior_i, prior_j, sigma_inv
                    )

                    estimand_i = np.where(
                        variance[i, j, :] != 0,
                        mu[i, j, :] + M3[i, j, :] / (6 * variance[i, j, :] * n),
                        mu[i, j, :],
                    )

                    estimand_j = np.where(
                        variance[ni, nj, :] != 0,
                        mu[ni, nj, :] + M3[ni, nj, :] / (6 * variance[ni, nj, :] * n),
                        mu[ni, nj, :],
                    )

                    estimand_i_variance = variance[i, j] / n
                    estimand_j_variance = variance[ni, nj] / n

                    wij = compute_w(
                        estimand_i,
                        estimand_j,
                        estimand_i_variance,
                        estimand_j_variance,
                    )

                    t = compute_t_statistic(wij)
                    mij = int(np.all(t < gamma_w))
                    kernels[i, j, di + radius, dj + radius] *= mij
                    memebership_funcion[i, j, di + radius, dj + radius] = mij

            sum_weights = np.sum(kernels[i, j, :, :])

            # Normalizamos cada peso de cada píxel j en el vecindario de i
            if sum_weights != 0:  # Evitar división por cero
                kernels[i, j, :, :] /= sum_weights

    return kernels, memebership_funcion


if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")
    samples = np.load("./samples.npy")
    # albedo:ch7-9 normales:ch10-12
    bitmap = mi.Bitmap("./aovs.exr")
    res = dict(bitmap.split())

    # Test a cada función por separado para comprobar que funcionan correctamente
    spp, mu, variance, M2, M3 = calculate_statistics(samples)
    gamma_w = calculate_critical_value(spp, spp)
    print("gamma_w = ", gamma_w)
    res = dict(bitmap.split())
    albedo = np.array(res["albedo"])
    normals = np.array(res["nn"])

    sigma = np.diag([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    kernels, memberships = denoiser(
        albedo, normals, spp, mu, variance, M2, M3, gamma_w, sigma
    )

    image = np.array(res["<root>"])
    result = apply_weights_to_image(image, kernels)

    bitmap = mi.Bitmap(result)

    bitmap.write("denoised_image.exr")

    print(kernels[217, 168])
    print(memberships[217, 168])
