import os
import sys

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <ground_truth.npy> <denoised.npy>")
        sys.exit(1)

    ground_truth_path = sys.argv[1]
    denoised_path = sys.argv[2]

    # Cargar los archivos
    ground_truth = np.load(ground_truth_path)
    denoised = np.load(denoised_path)

    # Calcular y mostrar MSE y RMSE
    print("RMSE denoised: ", np.sqrt(np.mean((ground_truth - denoised) ** 2)))
    print("MSE denoised:  ", np.mean((ground_truth - denoised) ** 2))
