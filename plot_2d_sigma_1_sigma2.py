import numpy as np
import matplotlib.pyplot as plt

def rotated_ellipsoid(x, theta=np.pi/6, lambdas=(1.0, 2.0)):
    """
    Evaluates a rotated ellipsoid function at an integer point x.
    
    The function is defined as:
      f(x) = λ₁*(x_rot[0])² + λ₂*(x_rot[1])²,
    where x_rot = R(θ) * x, and R(θ) is the rotation matrix.
    
    Parameters:
      x       : array-like, expected to have integer components.
      theta   : rotation angle in radians.
      lambdas : tuple with eigenvalues (scaling factors). Here we use (1.0, 2.0)
                for a moderate anisotropy.
    """
    x = np.array(x, dtype=int)  # ensure integer input
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    x_rot = R.dot(x)
    return lambdas[0] * x_rot[0]**2 + lambdas[1] * x_rot[1]**2

def sample_double_geometric(sigma, size=None):
    """
    Samples integer offsets from a symmetric double geometric (discrete Laplace) distribution.
    
    Parameters:
      sigma : mutation scale parameter.
      size  : number of samples to draw (if None, returns a single sample).
      
    Returns:
      An integer (or array of integers) sampled from the distribution.
    """
    p = np.exp(-1.0/sigma)
    u = np.random.rand() if size is None else np.random.rand(*np.atleast_1d(size))
    threshold = (1-p)/(1+p)
    if size is None:
        if u < threshold:
            return 0
        else:
            n = np.random.geometric(p=1-p)
            return np.random.choice([-1, 1]) * n
    else:
        u = np.asarray(u)
        samples = np.zeros(u.shape, dtype=int)
        idx = (u >= threshold)
        if np.any(idx):
            n = np.random.geometric(p=1-p, size=np.sum(idx))
            samples[idx] = np.random.choice([-1, 1], size=np.sum(idx)) * n
        return samples

def compute_progress_rate(x, sigma1, sigma2, num_samples=5000, theta=np.pi/6,
                          lambdas=(1.0, 2.0), lower_bound=-12, upper_bound=12):
    """
    Estimates the expected progress rate for a given parent x using mutation scales sigma1 and sigma2.
    
    Progress is defined as:
      Δf = max{0, f(x_parent) - f(x_offspring)}
    
    Offspring are generated by adding mutations (sampled via the double geometric distribution)
    to each coordinate of x, with the result truncated to lie within [lower_bound, upper_bound].
    """
    x = np.array(x, dtype=int)
    f_parent = rotated_ellipsoid(x, theta, lambdas)
    
    # Sample mutations for each coordinate
    delta1 = sample_double_geometric(sigma1, size=num_samples)
    delta2 = sample_double_geometric(sigma2, size=num_samples)
    
    # Generate offspring and clip to allowed domain
    offspring = np.vstack((x[0] + delta1, x[1] + delta2)).T
    offspring = np.clip(offspring, lower_bound, upper_bound)
    
    # Evaluate offspring and compute improvements
    f_offspring = np.array([rotated_ellipsoid(o, theta, lambdas) for o in offspring])
    improvements = f_parent - f_offspring
    improvements[improvements < 0] = 0
    return np.mean(improvements)

# -------------------------------
# Tuning Procedure
# -------------------------------

# Define the parent point (current solution) as (5,5)
x_parent = np.array([5, 5], dtype=int)

# Define grid for sigma parameters
# Extend the range to approximately 0.1 to 10 (avoiding sigma=0)
sigma_range = np.linspace(0.1, 10.0, 20)
progress_matrix = np.zeros((len(sigma_range), len(sigma_range)))

# Evaluate the progress rate for each (sigma1, sigma2) combination
for i, sigma1 in enumerate(sigma_range):
    for j, sigma2 in enumerate(sigma_range):
        progress_matrix[i, j] = compute_progress_rate(x_parent, sigma1, sigma2, num_samples=5000)

# Identify the optimum sigma parameters (maximizing expected progress)
max_idx = np.unravel_index(np.argmax(progress_matrix), progress_matrix.shape)
opt_sigma1 = sigma_range[max_idx[0]]
opt_sigma2 = sigma_range[max_idx[1]]
print("Optimal sigma1:", opt_sigma1, "Optimal sigma2:", opt_sigma2)

# -------------------------------
# Visualization of the Tuning Landscape
# -------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(progress_matrix, extent=[sigma_range[0], sigma_range[-1],
                                    sigma_range[0], sigma_range[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.xlabel(r'$\sigma_2$')
plt.ylabel(r'$\sigma_1$')
plt.title('Progress Rate Landscape')
plt.colorbar(label='Progress Rate')

# Mark the optimum
plt.scatter([opt_sigma2], [opt_sigma1], color='red', marker='x', s=100, label='Optimum')
plt.legend()
plt.show()
