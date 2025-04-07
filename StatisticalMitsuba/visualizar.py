import matplotlib.pyplot as plt
import numpy as np


def plot_pixel_distribution(x, y, data, data_box_cox):
    """
    Dibuja la distribución de los samples de un píxel (x, y) en los 3 canales RGB,
    y muestra la media (color final resultante) en ese píxel.

    Parámetros:
        - x, y: Coordenadas del píxel a analizar.
        - data: Array numpy con los samples originales (shape: (3, height, width, spp)).
        - data_box_cox: Array numpy con los samples después de la transformación Box-Cox.
    """
    # Extraer los samples del píxel en los tres canales (ahora el primer eje es RGB)
    samples_r = data[0, y, x, :]
    samples_g = data[1, y, x, :]
    samples_b = data[2, y, x, :]

    samples_r_bc = data_box_cox[0, y, x, :]
    samples_g_bc = data_box_cox[1, y, x, :]
    samples_b_bc = data_box_cox[2, y, x, :]

    # Calcular la media (color final)
    mean_r = np.mean(samples_r)
    mean_g = np.mean(samples_g)
    mean_b = np.mean(samples_b)

    print(
        f"Color RGB final en el píxel ({x}, {y}): ({mean_r:.4f}, {mean_g:.4f}, {mean_b:.4f})"
    )

    # Configuración de la gráfica
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    # Etiquetas y colores
    channels = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

    # Dibujar histogramas antes y después de Box-Cox
    for i, (samples, samples_bc, ax1, ax2) in enumerate(
        zip(
            [samples_r, samples_g, samples_b],
            [samples_r_bc, samples_g_bc, samples_b_bc],
            axes[:, 0],
            axes[:, 1],
        )
    ):
        ax1.hist(samples, bins=200, color=colors[i], alpha=0.7, density=True)
        ax1.axvline(np.mean(samples), color="black", linestyle="dashed", linewidth=1.5)
        ax1.set_title(f"{channels[i]} Channel - Original (Pixel {x}, {y})")
        ax1.set_xlabel("Sample Value")
        ax1.set_ylabel("Density")

        ax2.hist(samples_bc, bins=50, color=colors[i], alpha=0.7, density=True)
        ax2.axvline(
            np.mean(samples_bc), color="black", linestyle="dashed", linewidth=1.5
        )
        ax2.set_title(f"{channels[i]} Channel - Box-Cox (Pixel {x}, {y})")
        ax2.set_xlabel("Sample Value")
        ax2.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


# Cargar datos desde archivos .npy
data = np.load("samples.npy")
data_box_cox = np.load("samples_box_cox.npy")
plot_pixel_distribution(0, 0, data, data_box_cox)
