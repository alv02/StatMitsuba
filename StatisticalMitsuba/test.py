import numpy as np
from denoiser import box_cox, calculate_statistics, evaluate_base_filter
from scipy.special import boxcox


def test_evaluate_base_filter():
    prior_i = np.array([15, 15, 0.2, 0.2, 0.2, 0, 0, 1])
    prior_j = np.array([16, 16, 0.3, 0.4, 0.2, 0, 0, 1])
    sigma = np.diag([10, 10, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    sigma_inv = np.linalg.inv(sigma)

    expected_value_1 = 0.25924026064589156
    expected_value_2 = 1.0

    result_1 = evaluate_base_filter(prior_i, prior_j, sigma_inv)
    result_2 = evaluate_base_filter(prior_i, prior_i, sigma_inv)

    assert np.isclose(
        result_1, expected_value_1, atol=1e-6
    ), f"Test 1 falló: {result_1} != {expected_value_1}"
    assert np.isclose(
        result_2, expected_value_2, atol=1e-6
    ), f"Test 2 falló: {result_2} != {expected_value_2}"

    print("Todos los tests pasaron correctamente.")


def test_calculate_statistics():
    channels, height, width, spp = 3, 2, 2, 10
    np.random.seed(42)  # Para reproducibilidad

    # Generar muestras aleatorias positivas (Box-Cox requiere valores positivos)
    samples = np.abs(np.random.randn(channels, height, width, spp)) + 1

    n, mu, variance, M2, M3 = calculate_statistics(samples)
    print(box_cox(samples))
    print("mu")
    print(mu)
    print("variance")
    print(variance)
    print("M2")
    print(M2)
    print("M3")
    print(M3)


if __name__ == "__main__":
    test_evaluate_base_filter()
    test_calculate_statistics()
