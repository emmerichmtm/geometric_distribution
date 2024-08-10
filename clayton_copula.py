import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, uniform
from scipy.optimize import fsolve


# Clayton Copula functions
def clayton_copula_sample(u1, theta):
    """ Generate a sample from the Clayton copula given u1 and theta """
    v = uniform.rvs(size=len(u1))
    u2 = np.array(
        [(u1[i] ** (-theta) * (v[i] ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta) for i in range(len(u1))])
    return u2


def clayton_theta_from_rho(rho):
    """ Approximate theta parameter from desired correlation rho """
    if rho == 0:
        return 0
    return 2 * rho / (1 - rho)


def apply_clayton_copula_to_geometric_differences_with_heatmap(p1, p2, rho, n=1000):
    G1_1 = geom.rvs(p1, size=n)
    G2_1 = geom.rvs(p2, size=n)
    G1_2 = geom.rvs(p1, size=n)
    G2_2 = geom.rvs(p2, size=n)

    U1_1 = geom.cdf(G1_1, p1)
    U1_2 = geom.cdf(G1_2, p1)
    U2_1 = geom.cdf(G2_1, p2)
    U2_2 = geom.cdf(G2_2, p2)

    # Apply Clayton copula to introduce correlation
    theta = clayton_theta_from_rho(rho)
    U1_prime_1 = clayton_copula_sample(U1_1, theta)
    U2_prime_2 = clayton_copula_sample(U1_2, theta)

    # Transform back to the geometric scale
    G1_prime_1 = geom.ppf(U1_prime_1, p1)
    G2_prime_2 = geom.ppf(U2_prime_2, p2)

    # Calculate differences
    Z1_prime = G1_prime_1 - G2_1
    Z2_prime = G1_prime_1 - G2_2

    return Z1_prime, Z2_prime


# Parameters
p1 = 0.1
p2 = 0.2
rho = 0.4


n = 10000

# Generate correlated data using the Clayton Copula
Z1_prime_clayton, Z2_prime_clayton = apply_clayton_copula_to_geometric_differences_with_heatmap(p1, p2, rho, n)

# Filter data to keep only within the range [-20, 20]
mask_clayton = (Z1_prime_clayton >= -20) & (Z1_prime_clayton <= 20) & (Z2_prime_clayton >= -20) & (
            Z2_prime_clayton <= 20)
Z1_prime_clayton = Z1_prime_clayton[mask_clayton]
Z2_prime_clayton = Z2_prime_clayton[mask_clayton]

# Reinforce the symmetry by mirroring the samples
#can you randomly mirror the samples by multiplication with -1
for i in range(len(Z1_prime_clayton)):
    if np.random.choice([True, False]):
        Z1_prime_clayton[i] = -Z1_prime_clayton[i]
        Z2_prime_clayton[i] = -Z2_prime_clayton[i]

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime_clayton, Z2_prime_clayton, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap of Bivariate Distribution (Clayton Copula, rho=0.4)')
plt.xlabel('Z1_prime')
plt.ylabel('Z2_prime')
plt.show()

# Plot the 3D histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create histogram data
hist, xedges, yedges = np.histogram2d(Z1_prime_clayton, Z2_prime_clayton, bins=30)

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax.set_title('3D Histogram of Bivariate Distribution (Clayton Copula, rho=0.4)')
ax.set_xlabel('Z1_prime')
ax.set_ylabel('Z2_prime')
ax.set_zlabel('Counts')

plt.show()
