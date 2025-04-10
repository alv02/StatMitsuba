import mitsuba as mi
import numpy as np
import torch


def evaluate_base_filter(prior_i, prior_j, sigma_inv):
    # Compute the difference (p_j - p_i)
    diff = prior_j - prior_i

    # Calculate the weight using the Gaussian falloff formula
    weight = np.exp(-0.5 * (diff.T @ sigma_inv @ diff))

    return weight


def denoiser(albedo, normals, radius=5):
    height, width, _ = np.shape(albedo)
    kernels = np.zeros((height, width, radius * 2 + 1, radius * 2 + 1))
    sigma = np.diag([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
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

            sum_weights = np.sum(kernels[i, j, :, :])

            # Normalizamos cada peso de cada píxel j en el vecindario de i
            if sum_weights != 0:  # Evitar división por cero
                kernels[i, j, :, :] /= sum_weights

    return kernels


if __name__ == "__main__":
    mi.set_variant("llvm_ad_rgb")
    samples = np.load("./samples.npy")
    # albedo:ch7-9 normales:ch10-12
    bitmap = mi.Bitmap("./aovs.exr")
    res = dict(bitmap.split())

    # Test a cada función por separado para comprobar que funcionan correctamente
    res = dict(bitmap.split())
    albedo = np.array(res["albedo"])
    normals = np.array(res["nn"])

    kernels = denoiser(albedo, normals, radius=20)

    print(kernels[217, 168])
