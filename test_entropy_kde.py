import numpy as np
import pytest
from scipy.stats import norm, uniform
from conditional_entropy import entropy_kde

# Mock `entropy_kde` function if importing it externally.
# from module import entropy_kde


def test_uniform_distribution():
    # Case 1: Uniform distribution on [0, 1]
    np.random.seed(42)
    data = np.random.uniform(0, 1, size=1000)

    # Theoretical entropy for Uniform(0, 1)
    expected_entropy = np.log(1 - 0)  # log(b-a) for a=0, b=1

    computed_entropy = entropy_kde(data, bandwidth=0.1)

    assert np.isclose(
        computed_entropy, expected_entropy, atol=0.1
    ), f"Expected {expected_entropy}, got {computed_entropy}"


def test_normal_distribution():
    # Case 2: Normal distribution N(0, 1)
    np.random.seed(42)
    mu = 0
    sigma = 1
    data = np.random.normal(mu, sigma, size=1000)

    # Theoretical entropy for Normal(0, 1)
    expected_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)

    computed_entropy = entropy_kde(data, bandwidth=0.2)

    assert np.isclose(
        computed_entropy, expected_entropy, atol=0.1
    ), f"Expected {expected_entropy}, got {computed_entropy}"


def test_multimodal_distribution():
    # Case 3: Mixture of two normals
    np.random.seed(42)
    data = np.concatenate(
        [np.random.normal(-2, 1, size=500), np.random.normal(2, 1, size=500)]
    )

    # Approximate the entropy for multimodal data (can be harder to validate)
    computed_entropy = entropy_kde(data, bandwidth=0.3)

    # Assert that the entropy is reasonable (not too low/high for multimodal)
    # Expected range can vary depending on the KDE approximation
    assert computed_entropy > 2.0, f"Entropy too low: {computed_entropy}"
    assert computed_entropy < 4.0, f"Entropy too high: {computed_entropy}"


if __name__ == "__main__":
    test_uniform_distribution()
    test_normal_distribution()
    test_multimodal_distribution()
