import os

import mitsuba as mi
import numpy as np
import scipy.stats as stats
from numba import njit, prange


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

    result = numerator / denominator_safe

    result = np.where((numerator == 0) & (denominator == 0), 0.5, result)

    variance_zero = (estimand_i_variance == 0) | (estimand_j_variance == 0)
    values_differ = estimand_i != estimand_j
    result = np.where(variance_zero & values_differ, np.full_like(numerator, 1), result)
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
    gamma_w,
    estimands,
    estimands_variance,
    sigma,
    radius=5,
):
    height, width, channels = np.shape(image)

    denoised_image = np.zeros_like(image)

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

                    estimand_i = estimands[i, j]

                    estimand_j = estimands[ni, nj]

                    estimand_i_variance = estimands_variance[i, j]
                    estimand_j_variance = estimands_variance[ni, nj]

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
    statistics = np.load("./staircase_stats.npy")
    # albedo:ch7-9 normales:ch10-12
    bitmap = mi.Bitmap("./staircase.exr")
    res = dict(bitmap.split())

    estimands = statistics[:, :, :, 0]
    estimands_variance = statistics[:, :, :, 1]
    estimands = np.transpose(estimands, (1, 2, 0))
    estimands_variance = np.transpose(estimands_variance, (1, 2, 0))
    spp = statistics[0, 0, 0, 2]
    # Test a cada función por separado para comprobar que funcionan correctamente
    res = dict(bitmap.split())
    albedo = np.array(res["albedo"])
    normals = np.array(res["nn"])
    image = np.array(res["<root>"])

    gamma_w = calculate_critical_value(spp, spp)
    sigma = np.diag([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    result = denoiser(
        image,
        albedo,
        normals,
        gamma_w,
        estimands,
        estimands_variance,
        sigma,
        radius=5,
    )

    bitmap = mi.Bitmap(result)

    bitmap.write("denoised_image.exr")
