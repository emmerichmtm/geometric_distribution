import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, norm


# Function to generate and symmetrize correlated geometric random variables
def generate_symmetric_correlated_geometric_differences(p1, p2, rho, n=1000):
    # Step 1: Generate independent normal variables X1, X2, and X3
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)

    # Step 2: Create correlated normal variables Y1, Y2 using X1 and X2, X3
    Y1 = rho * X1 + np.sqrt(1 - rho ** 2) * X2
    Y2 = rho * X1 + np.sqrt(1 - rho ** 2) * X3  # Ensure Y2 is symmetrically generated

    # Step 3: Transform normal variables to uniform
    U1 = norm.cdf(Y1)
    U2 = norm.cdf(Y2)

    # Step 4: Transform uniform variables to geometric
    G1_1 = geom.ppf(U1, p1)
    G1_2 = geom.ppf(U2, p1)
    G2_1 = geom.rvs(p2, size=n)
    G2_2 = geom.rvs(p2, size=n)

    # Step 5: Calculate the differences
    Z1_prime = G1_1 - G2_1
    Z2_prime = G1_2 - G2_2

    # Step 6: Randomly flip signs with probability 0.5
    flip_sign = np.random.choice([-1, 1], size=n)
    Z1_prime *= flip_sign
    Z2_prime *= flip_sign

    return Z1_prime, Z2_prime


# Parameters
p1 = 0.1
p2 = 0.3
rho = 0.9
n = 10000

# Generate the correlated differences with sign flipping
Z1_prime, Z2_prime = generate_symmetric_correlated_geometric_differences(p1, p2, rho, n)

# Filter data to keep only within the range [-20, 20]
mask = (Z1_prime >= -20) & (Z1_prime <= 20) & (Z2_prime >= -20) & (Z2_prime <= 20)
Z1_prime = Z1_prime[mask]
Z2_prime = Z2_prime[mask]

# Plot the heatmap
plt.figure(figsize=(8, 6))
heatmap_data, xedges, yedges = np.histogram2d(Z1_prime, Z2_prime, bins=30)
sns.heatmap(heatmap_data.T, cmap='Blues', xticklabels=np.round(xedges, 2), yticklabels=np.round(yedges, 2))
plt.title('Heatmap of Bivariate Distribution (ρ = 0.4)')
plt.xlabel('Z1_prime')
plt.ylabel('Z2_prime')
plt.show()

# Plot the 3D histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create histogram data
hist, xedges, yedges = np.histogram2d(Z1_prime, Z2_prime, bins=30)

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

ax.set_title('3D Histogram of Bivariate Distribution (ρ = 0.4)')
ax.set_xlabel('Z1_prime')
ax.set_ylabel('Z2_prime')
ax.set_zlabel('Counts')

plt.show()
