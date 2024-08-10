import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, norm, uniform
from scipy.optimize import fsolve

# Clayton Copula functions
def clayton_copula_sample(u1, theta):
    """ Generate a sample from the Clayton copula given u1 and theta """
    v = uniform.rvs(size=len(u1))
    u2 = np.array([(u1[i]**(-theta) * (v[i]**(-theta / (theta + 1)) - 1) + 1)**(-1 / theta) for i in range(len(u1))])
    return u2

def clayton_theta_from_rho(rho):
    """ Approximate theta parameter from desired correlation rho """
    if rho == 0:
        return 0
    return 2 * rho / (1 - rho)

# Frank Copula functions
def frank_copula_sample(u1, theta):
    """ Generate a sample from Frank copula given u1 and theta """
    def frank_copula_inverse(u1, v, theta):
        return -np.log(1 + (np.exp(-theta * u1) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)) / theta

    v = uniform.rvs(size=len(u1))
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

# Function to apply a copula and compute differences
def apply_copula_to_geometric_differences(copula_type, p1, p2, rho, n=1000):
    if copula_type == 'gaussian':
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        gaussian_samples = np.random.multivariate_normal(mean, cov, n)
        U1 = norm.cdf(gaussian_samples[:, 0])
        U2 = norm.cdf(gaussian_samples[:, 1])
    elif copula_type == 'frank':
        G1 = geom.rvs(p1, size=n)
        G2 = geom.rvs(p2, size=n)
        U1 = geom.cdf(G1, p1)
        U2 = geom.cdf(G2, p2)
        theta = frank_theta_from_rho(rho)
        U1 = frank_copula_sample(U1, theta)
        U2 = frank_copula_sample(U2, theta)
    elif copula_type == 'clayton':
        G1 = geom.rvs(p1, size=n)
        G2 = geom.rvs(p2, size=n)
        U1 = geom.cdf(G1, p1)
        U2 = geom.cdf(G2, p2)
        theta = clayton_theta_from_rho(rho)
        U1 = clayton_copula_sample(U1, theta)
        U2 = clayton_copula_sample(U2, theta)
    else:
        raise ValueError("Invalid copula type specified.")

    G1_prime = geom.ppf(U1, p1)
    G2_prime = geom.ppf(U2, p2)
    Z1_prime = G1_prime - G2_prime
    Z2_prime = G1_prime - G2_prime

    return Z1_prime, Z2_prime

# Parameters
p1 = 0.1
p2 = 0.2
rho = 0.4
n = 10000

# Generate data for each copula
Z1_prime_gaussian, Z2_prime_gaussian = apply_copula_to_geometric_differences('gaussian', p1, p2, rho, n)
Z1_prime_frank, Z2_prime_frank = apply_copula_to_geometric_differences('frank', p1, p2, rho, n)
Z1_prime_clayton, Z2_prime_clayton = apply_copula_to_geometric_differences('clayton', p1, p2, rho, n)

# Filter data to keep only within the range [-20, 20]
mask_gaussian = (Z1_prime_gaussian >= -20) & (Z1_prime_gaussian <= 20) & (Z2_prime_gaussian >= -20) & (Z2_prime_gaussian <= 20)
Z1_prime_gaussian = Z1_prime_gaussian[mask_gaussian]
Z2_prime_gaussian = Z2_prime_gaussian[mask_gaussian]

mask_frank = (Z1_prime_frank >= -20) & (Z1_prime_frank <= 20) & (Z2_prime_frank >= -20) & (Z2_prime_frank <= 20)
Z1_prime_frank = Z1_prime_frank[mask_frank]
Z2_prime_frank = Z2_prime_frank[mask_frank]

mask_clayton = (Z1_prime_clayton >= -20) & (Z1_prime_clayton <= 20) & (Z2_prime_clayton >= -20) & (Z2_prime_clayton <= 20)
Z1_prime_clayton = Z1_prime_clayton[mask_clayton]
Z2_prime_clayton = Z2_prime_clayton[mask_clayton]

# Plot the heatmaps for each copula
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime_gaussian, Z2_prime_gaussian, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap (Gaussian Copula)')
plt.xlabel('Z1_prime')
plt.ylabel('Z2_prime')

plt.subplot(1, 3, 2)
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime_frank, Z2_prime_frank, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap (Frank Copula)')
plt.xlabel('Z1_prime')
plt.ylabel('Z2_prime')

plt.subplot(1, 3, 3)
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime_clayton, Z2_prime_clayton, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap (Clayton Copula)')
plt.xlabel('Z1_prime')
plt.ylabel('Z2_prime')

plt.show()

# Plot the 3D histograms for each copula
fig = plt.figure(figsize=(18, 6))

# Gaussian Copula
ax = fig.add_subplot(131, projection='3d')
hist, xedges, yedges = np.histogram2d(Z1_prime_gaussian, Z2_prime_gaussian, bins=30)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_title('3D Histogram (Gaussian Copula)')
ax.set_xlabel('Z1_prime')
ax.set_ylabel('Z2_prime')
ax.set_zlabel('Counts')

# Frank Copula
ax = fig.add_subplot(132, projection='3d')
hist, xedges, yedges = np.histogram2d(Z1_prime_frank, Z2_prime_frank, bins=30)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_title('3D Histogram (Frank Copula)')
ax.set_xlabel('Z1_prime')
ax.set_ylabel('Z2_prime')
ax.set_zlabel('Counts')
