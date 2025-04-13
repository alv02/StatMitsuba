import os

import mitsuba as mi
import numpy as np
import scipy.stats as stats
from numba import njit, prange


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


@njit(parallel=True)
def denoiser(
    image,
    albedo,
    normals,
    n,
    mu,
    variance,
    M2,
    M3,
    gamma_w,
    sigma,
    radius=5,
    epsilon=1e-8,
):
    height, width, channels = np.shape(image)

    denoised_image = np.zeros_like(image)

    variance = np.where(variance == 0, epsilon, variance)
    sigma_inv = np.linalg.inv(sigma)

    # Pyxel i
    for i in prange(height):
        for j in range(width):
            kernel = np.zeros((radius * 2 + 1, radius * 2 + 1))
            memebership_function = np.zeros((radius * 2 + 1, radius * 2 + 1))
            weight_optimal = np.zeros((radius * 2 + 1, radius * 2 + 1, channels))
            t_statistical = np.zeros((radius * 2 + 1, radius * 2 + 1, channels))

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
                        kernel[di + radius, dj + radius] = 1
                        memebership_function[di + radius, dj + radius] = 1
                        continue

                    if not (0 <= ni < height and 0 <= nj < width):
                        continue
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

                    kernel[di + radius, dj + radius] = evaluate_base_filter(
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
                    kernel[di + radius, dj + radius] *= mij
                    memebership_function[di + radius, dj + radius] = mij
                    weight_optimal[di + radius, dj + radius] = wij
                    t_statistical[di + radius, dj + radius] = t

            sum_weights = np.sum(kernel[:, :])

            if sum_weights != 0:
                kernel[:, :] /= sum_weights

            # Aplicar el kernel al pixel i
            weighted_sum = np.zeros(channels)
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        weighted_sum += (
                            kernel[di + radius, dj + radius] * image[ni, nj, :]
                        )

            denoised_image[i, j, :] = weighted_sum

    return denoised_image


if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")
    samples = np.load("./staircase_samples.npy")
    # albedo:ch7-9 normales:ch10-12
    bitmap = mi.Bitmap("./staircase.exr")
    res = dict(bitmap.split())

    # Test a cada función por separado para comprobar que funcionan correctamente
    spp, mu, variance, M2, M3 = calculate_statistics(samples)
    gamma_w = calculate_critical_value(spp, spp)
    res = dict(bitmap.split())
    albedo = np.array(res["albedo"])
    normals = np.array(res["nn"])
    image = np.array(res["<root>"])
    non_zero_variances = variance[variance > 0]

    sigma = np.diag([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    result = denoiser(
        image, albedo, normals, spp, mu, variance, M2, M3, gamma_w, sigma, radius=5
    )

    bitmap = mi.Bitmap(result)

    bitmap.write("denoised_image.exr")
