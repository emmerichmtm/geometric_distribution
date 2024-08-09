import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr

def simulate_correlated_bernoulli(N, p1, p2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, N)

    bernoulli_samples = np.zeros_like(normal_samples)
    bernoulli_samples[:, 0] = (normal_samples[:, 0] < np.percentile(normal_samples[:, 0], p1 * 100)).astype(int)
    bernoulli_samples[:, 1] = (normal_samples[:, 1] < np.percentile(normal_samples[:, 1], p2 * 100)).astype(int)

    df = pd.DataFrame(bernoulli_samples, columns=['Bernoulli_Variable_1', 'Bernoulli_Variable_2'])

    return df

def compute_first_success_times(df):
    c1 = 0
    c2 = 0
    found_X1 = False
    found_X2 = False

    for i in range(len(df)):
        if df['Bernoulli_Variable_1'][i] == 1 and not found_X1:
            found_X1 = True
        else:
            if not found_X1:
                c1 += np.random.choice([-1, 1])

        if df['Bernoulli_Variable_2'][i] == 1 and not found_X2:
            found_X2 = True
        else:
            if not found_X2:
                c2 += np.random.choice([-1, 1])

    return c1, c2

# Parameters
N = 50  # Number of samples
p1 = 0.01  # Probability of success for the first Bernoulli variable
p2 = 0.08  # Probability of success for the second Bernoulli variable
rho = 0.9 # Correlation coefficient

num_repetitions = 10000
results = []

for _ in range(num_repetitions):
    df_samples = simulate_correlated_bernoulli(N, p1, p2, rho)
    final_c1, final_c2 = compute_first_success_times(df_samples)
    results.append((final_c1, final_c2))

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results, columns=['Final_C1', 'Final_C2'])

# Plot heatmap of the counters after random walk
plt.figure(figsize=(10, 10))
heatmap_data = pd.crosstab(results_df['Final_C1'], results_df['Final_C2'])
sns.heatmap(heatmap_data, annot=False, cmap="viridis")
plt.title('Heatmap of Final Counters C1 and C2 After Random Walk')
plt.xlabel('Final C2 Counter')
plt.ylabel('Final C1 Counter')
plt.show()

# Create a 3D plot of the results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare data for 3D plotting
hist, xedges, yedges = np.histogram2d(results_df['Final_C1'], results_df['Final_C2'], bins=50)

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Plot bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', cmap='viridis')

# Set labels
ax.set_xlabel('Final C1 Counter')
ax.set_ylabel('Final C2 Counter')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram of Final Counters C1 and C2')

plt.show()
