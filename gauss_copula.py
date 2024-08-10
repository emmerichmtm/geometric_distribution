import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, norm

# Define the Gaussian Copula for two geometric differences
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
p1 = 0.05
p2 = 0.1
rho = 0.8
n = 10000

# Generate the data
Z1_prime_gauss, Z2_prime_gauss = apply_gaussian_copula_to_geometric_differences(p1, p2, rho, n)

# Filter data to keep only within the range [-20, 20]
mask = (Z1_prime_gauss >= -20) & (Z1_prime_gauss <= 20) & (Z2_prime_gauss >= -20) & (Z2_prime_gauss <= 20)
Z1_prime_gauss = Z1_prime_gauss[mask]
Z2_prime_gauss = Z2_prime_gauss[mask]

# Create a heatmap
plt.figure(figsize=(8, 6))
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime_gauss, Z2_prime_gauss, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap of Bivariate Distribution (Filtered)')
plt.xlabel('Z1_prime (Gaussian Copula)')
plt.ylabel('Z2_prime (Gaussian Copula)')
plt.show()

# Create a 3D histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create histogram data
hist, xedges, yedges = np.histogram2d(Z1_prime_gauss, Z2_prime_gauss, bins=30)

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax.set_title('3D Histogram of Bivariate Distribution (Filtered)')
ax.set_xlabel('Z1_prime (Gaussian Copula)')
ax.set_ylabel('Z2_prime (Gaussian Copula)')
ax.set_zlabel('Counts')

plt.show()
