import numpy as np
import pandas as pd


# Manhattan distance function (fitness function)
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


# Simulate correlated Bernoulli process to generate c1 and c2
def simulate_bernoulli_walk(N, p1, p2):
    df_samples = simulate_correlated_bernoulli(N, p1, p2, rho=0.0)
    c1, c2 = compute_first_success_times(df_samples)
    return c1, c2


# Updated mutation operator with self-adaptive step-size control
def mutate(x1, x2, p1, p2, N):
    # Update step-sizes
    new_p1 = p1 * np.exp(np.random.normal(0, 0.01))
    new_p2 = p2 * np.exp(np.random.normal(0, 0.01))

    # Simulate Bernoulli walk to generate c1 and c2 using the new step-sizes
    c1, c2 = simulate_bernoulli_walk(N, new_p1, new_p2)

    # Mutate x1 and x2 using the generated c1 and c2
    new_x1 = x1 + c1
    new_x2 = x2 + c2

    return new_x1, new_x2, new_p1, new_p2


# Evolution strategy (μ, λ)
def evolution_strategy(mu, lambda_, num_generations, N):
    # Initialize population
    population = [{'x1': np.random.randint(-100, 101),
                   'x2': np.random.randint(-100, 101),
                   'p1': np.random.uniform(0.01, 0.1),
                   'p2': np.random.uniform(0.01, 0.1),
                   'fitness': None}
                  for _ in range(mu)]

    for generation in range(num_generations):
        # Evaluate fitness for the initial population or when fitness is not computed
        for individual in population:
            if individual['fitness'] is None:
                individual['fitness'] = manhattan_distance(individual['x1'], individual['x2'], 0, 0)

        # Sort population by fitness (minimization)
        population = sorted(population, key=lambda x: x['fitness'])

        # Select the top mu individuals
        selected_population = population[:mu]

        # Generate lambda_ offspring by mutating the selected population
        offspring = []
        for _ in range(lambda_):
            parent = np.random.choice(selected_population)
            new_x1, new_x2, new_p1, new_p2 = mutate(parent['x1'], parent['x2'], parent['p1'], parent['p2'], N)
            # Calculate the fitness of the offspring immediately after mutation
            offspring_fitness = manhattan_distance(new_x1, new_x2, 0, 0)
            offspring.append({'x1': new_x1, 'x2': new_x2, 'p1': new_p1, 'p2': new_p2, 'fitness': offspring_fitness})

        # Replace the current population with the offspring
        population = offspring

        # Print the best solution of the current generation
        best_individual = min(population, key=lambda x: x['fitness'])
        print(f"Generation {generation + 1}: Best Fitness = {best_individual['fitness']}, "
              f"x1 = {best_individual['x1']}, x2 = {best_individual['x2']}, "
              f"p1 = {best_individual['p1']:.5f}, p2 = {best_individual['p2']:.5f}")

    # Return the best solution found
    best_individual = min(population, key=lambda x: x['fitness'])
    return best_individual


# Simulate correlated Bernoulli process
def simulate_correlated_bernoulli(N, p1, p2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_samples = np.random.multivariate_normal(mean, cov, N)

    bernoulli_samples = np.zeros_like(normal_samples)
    bernoulli_samples[:, 0] = (normal_samples[:, 0] < np.percentile(normal_samples[:, 0], p1 * 100)).astype(int)
    bernoulli_samples[:, 1] = (normal_samples[:, 1] < np.percentile(normal_samples[:, 1], p2 * 100)).astype(int)

    df = pd.DataFrame(bernoulli_samples, columns=['Bernoulli_Variable_1', 'Bernoulli_Variable_2'])

    return df


# Compute the first success times c1 and c2
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


# Main execution
if __name__ == "__main__":
    print("Running Evolution Strategy with Self-Adaptive Mutation...")
    best_solution = evolution_strategy(mu=10, lambda_=50, num_generations=100, N=50)
    print(f"Best Solution: {best_solution}")
