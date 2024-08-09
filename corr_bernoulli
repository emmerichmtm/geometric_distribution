import numpy as np
from scipy.stats import norm

def correlated_bernoulli(correlation_matrix, p):
    """
    Generate correlated Bernoulli samples given a correlation matrix and Bernoulli probabilities.
    
    Parameters:
    - correlation_matrix: A positive-definite correlation matrix (numpy array).
    - p: A vector of Bernoulli probabilities (list or numpy array).
    
    Returns:
    - samples: A vector of correlated Bernoulli samples (numpy array).
    """
    # Step 1: Perform Cholesky decomposition on the correlation matrix
    L = np.linalg.cholesky(correlation_matrix)
    
    # Step 2: Generate independent standard normal samples
    z = np.random.normal(size=len(p))
    
    # Step 3: Induce correlation by multiplying with the Cholesky factor
    x = L @ z
    
    # Step 4: Convert to correlated Bernoulli samples
    samples = (norm.cdf(x) < p).astype(int)
    
    return samples

# Example usage
if __name__ == "__main__":
    # Example correlation matrix (3x3, symmetric, positive definite)
    correlation_matrix = np.array([[1.0, 0.5, 0.2],
                                   [0.5, 1.0, 0.3],
                                   [0.2, 0.3, 1.0]])

    # Example Bernoulli probabilities
    p = np.array([0.7, 0.5, 0.6])

    # Generate correlated Bernoulli samples
    samples = correlated_bernoulli(correlation_matrix, p)
    print("Correlated Bernoulli Samples:", samples)
