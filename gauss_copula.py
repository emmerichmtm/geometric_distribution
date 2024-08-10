import numpy as np
from scipy.stats import geom, norm

def apply_gaussian_copula_to_geometric_differences(p1, p2, rho, n=1000):
    # Generate correlated Gaussian variables for two pairs of differences
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    gaussian_samples_1 = np.random.multivariate_normal(mean, cov, n)
    gaussian_samples_2 = np.random.multivariate_normal(mean, cov, n)

    # Transform to uniform
    U1_1 = norm.cdf(gaussian_samples_1[:, 0])
    U1_2 = norm.cdf(gaussian_samples_1[:, 1])
    U2_1 = norm.cdf(gaussian_samples_2[:, 0])
    U2_2 = norm.cdf(gaussian_samples_2[:, 1])

    # Transform to geometric variables
    G1_1 = geom.ppf(U1_1, p1)
    G2_1 = geom.ppf(U2_1, p2)
    G1_2 = geom.ppf(U1_2, p1)
    G2_2 = geom.ppf(U2_2, p2)

    # Calculate differences
    Z1_prime = G1_1 - G2_1
    Z2_prime = G1_2 - G2_2

    return Z1_prime, Z2_prime

# Parameters
p1 = 0.5  # Success probability for the first geometric distribution
p2 = 0.7  # Success probability for the second geometric distribution
rho = 0.4  # Desired correlation between the differences
n = 10000  # Sample size

# Apply Gaussian Copula
Z1_prime_gauss, Z2_prime_gauss = apply_gaussian_copula_to_geometric_differences(p1, p2, rho, n)

# Calculate empirical correlation
empirical_rho_gauss = np.corrcoef(Z1_prime_gauss, Z2_prime_gauss)[0, 1]

# Output Results
print(f"Empirical correlation (Gaussian Copula): {empirical_rho_gauss}")
