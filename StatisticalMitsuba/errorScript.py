import csv
import os
import sys

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Uso: {sys.argv[0]} <transformacion> <ground_truth.npy> <denoised.npy>")
        sys.exit(1)

    transform = sys.argv[1]
    ground_truth_path = sys.argv[2]
    denoised_path = sys.argv[3]

    filename = os.path.basename(denoised_path)
    parts = filename.replace(".npy", "").split("_")

    try:
        lambda_idx = parts.index("lambda") + 1
        alpha_idx = parts.index("alpha") + 1
        lambda_val = parts[lambda_idx]
        alpha_val = parts[alpha_idx]
    except Exception as e:
        print(f"Error al extraer lambda/alpha del nombre: {filename}")
        sys.exit(1)

    # Cargar datos
    ground_truth = np.load(ground_truth_path)
    denoised = np.load(denoised_path)

    # Calcular métricas
    mse = np.mean((ground_truth - denoised) ** 2)
    rmse = np.sqrt(mse)

    # Guardar en CSV
    output_csv = "metrics.csv"
    write_header = not os.path.exists(output_csv)
    with open(output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["transformacion", "lambda", "alpha", "rmse", "mse"])
        writer.writerow([transform, lambda_val, alpha_val, rmse, mse])
