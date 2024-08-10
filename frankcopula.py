import numpy as np
from scipy.stats import geom, uniform
from scipy.optimize import fsolve


def frank_copula_sample(u1, theta):
    """ Generate a sample from Frank copula given u1 and theta """

    def frank_copula_inverse(u1, v, theta):
        # Frank copula inverse function to solve for v
        return -np.log(1 + (np.exp(-theta * u1) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)) / theta

    v = uniform.rvs(size=len(u1))  # generate independent uniform variables
    u2 = np.array([fsolve(frank_copula_inverse, 0.5, args=(u1[i], theta))[0] for i in range(len(u1))])
    return u2


def frank_theta_from_rho(rho):
    """ Approximate theta parameter from desired correlation rho """
    if rho == 0:
        return 0
    if rho > 0:
        return -np.log(rho) / (1 - rho)
    else:
        return np.log(-rho) / (1 + rho)


def apply_frank_copula_to_geometric_difference(p1, p2, rho, n=1000):
    # Step 1: Simulate independent geometric differences
    G1_1 = geom.rvs(p1, size=n)
    G2_1 = geom.rvs(p2, size=n)
    G1_2 = geom.rvs(p1, size=n)
    G2_2 = geom.rvs(p2, size=n)

    Z1 = G1_1 - G2_1
    Z2 = G1_2 - G2_2

    print(f"Z1 variance: {np.var(Z1)}, Z2 variance: {np.var(Z2)}")

    # Step 2: Transform to uniform variables using empirical CDF
    U1 = np.array([np.mean(Z1 <= z) for z in Z1])
    U2 = np.array([np.mean(Z2 <= z) for z in Z2])

    # Step 3: Apply Frank copula to introduce correlation
    theta = frank_theta_from_rho(rho)
    U2_prime = frank_copula_sample(U1, theta)

    # Clamp the values to the range [0, 1]
    U1 = np.clip(U1, 0, 1)
    U2_prime = np.clip(U2_prime, 0, 1)

    # Step 4: Transform back to the original scale using inverse empirical CDF
    Z1_prime = np.percentile(Z1, U1 * 100)
    Z2_prime = np.percentile(Z2, U2_prime * 100)

    print(f"Z1_prime variance: {np.var(Z1_prime)}, Z2_prime variance: {np.var(Z2_prime)}")

    return Z1_prime, Z2_prime


# Example usage:
p1 = 0.5  # success probability for the first geometric distribution
p2 = 0.7  # success probability for the second geometric distribution
rho = -0.9  # desired correlation between the differences

Z1_prime, Z2_prime = apply_frank_copula_to_geometric_difference(p1, p2, rho, n=10000)

# Check the empirical correlation
if np.var(Z1_prime) == 0 or np.var(Z2_prime) == 0:
    print("One of the distributions has zero variance; cannot compute correlation.")
else:
    empirical_rho = np.corrcoef(Z1_prime, Z2_prime)[0, 1]
    print(f"Empirical correlation: {empirical_rho}")
