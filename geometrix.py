import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def simulate_correlated_bernoulli(N, p1, p2, rho):
    # Generate correlated standard normal variables
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, N)

    # Transform the normal variables into uniform variables using the CDF of the normal distribution
    uniform_samples = np.random.normal(size=(N, 2))
    uniform_samples[:, 0] = (normal_samples[:, 0] - np.mean(normal_samples[:, 0])) / np.std(normal_samples[:, 0])
    uniform_samples[:, 1] = (normal_samples[:, 1] - np.mean(normal_samples[:, 1])) / np.std(normal_samples[:, 1])

    # Convert uniform samples to Bernoulli samples using the inverse CDF (percent point function) of the Bernoulli distribution
    bernoulli_samples = np.zeros_like(uniform_samples)
    bernoulli_samples[:, 0] = (uniform_samples[:, 0] < np.percentile(uniform_samples[:, 0], p1 * 100)).astype(int)
    bernoulli_samples[:, 1] = (uniform_samples[:, 1] < np.percentile(uniform_samples[:, 1], p2 * 100)).astype(int)

    # Create a DataFrame for easy visualization
    df = pd.DataFrame(bernoulli_samples, columns=['Bernoulli_Variable_1', 'Bernoulli_Variable_2'])

    return df

def compute_first_success_times(df):
    # Find the index of the first success (1) in each column
    first_success_X1 = df['Bernoulli_Variable_1'].eq(1).idxmax() + 1
    first_success_X2 = df['Bernoulli_Variable_2'].eq(1).idxmax() + 1
    return first_success_X1, first_success_X2

# Parameters
N = 1000  # Number of samples
p1 = 0.3  # Probability of success for the first Bernoulli variable
p2 = 0.7  # Probability of success for the second Bernoulli variable
rho = 0.5  # Correlation coefficient

# Number of repetitions for statistics
num_repetitions = 1000

# Store results
results = []

for _ in range(num_repetitions):
    # Generate samples
    df_samples = simulate_correlated_bernoulli(N, p1, p2, rho)
    # Compute time of first success
    first_success_times = compute_first_success_times(df_samples)
    results.append(first_success_times)

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results, columns=['First_Success_X1', 'First_Success_X2'])

# Compute Pearson correlation coefficient
correlation, p_value = pearsonr(results_df['First_Success_X1'], results_df['First_Success_X2'])

# Display the Pearson correlation coefficient
print(f"Pearson Correlation Coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Compute statistics
stats = results_df.describe()

# Display the statistics
print("First Success Times Statistics:")
print(stats)

# Plot heatmap of the time to first success
plt.figure(figsize=(10, 8))
heatmap_data = pd.crosstab(results_df['First_Success_X1'], results_df['First_Success_X2'])
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
plt.title('Heatmap of Time to First Success for X1 and X2')
plt.xlabel('Time to First Success for X2')
plt.ylabel('Time to First Success for X1')
plt.show()
