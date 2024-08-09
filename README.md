# geometric_distribution
+----------------------------------------------------------------------------------------+
|                                        README.md                                       |
+----------------------------------------------------------------------------------------+
| # Correlated Bernoulli Random Walk Simulation                                          |
|                                                                                        |
| This project simulates a 1-D random walk driven by correlated Bernoulli trials.        |
| It generates and analyzes the distribution of counters after a random walk process     |
| influenced by two Bernoulli variables. The results are visualized using heatmaps to    |
| show the relationship between the two counters. This simulation serves as a variant    |
| of the zero-centered geometric distribution, where the counters can both increase      |
| and decrease.                                                                          |
|                                                                                        |
| ## Overview                                                                            |
|                                                                                        |
| The simulation consists of the following steps:                                        |
|                                                                                        |
| 1. **Generation of Correlated Bernoulli Variables**:                                   |
|    - Two correlated Bernoulli variables, `X1` and `X2`, are generated using specified  |
|      probabilities `p1` and `p2` and a correlation coefficient `rho`.                 |
|                                                                                        |
| 2. **1-D Random Walk**:                                                                |
|    - For each Bernoulli variable, a random walk is performed until the first success   |
|      (i.e., when the variable equals 1) is observed. The counters `c1` and `c2` track  |
|      the progress of the random walk.                                                  |
|                                                                                        |
| 3. **Analysis and Visualization**:                                                     |
|    - After multiple repetitions, the final values of the counters `c1` and `c2` are    |
|      analyzed and visualized using heatmaps to understand their distribution.          |
|                                                                                        |
| ## How to Set Parameters                                                               |
|                                                                                        |
| ### `p1` and `p2`                                                                      |
|                                                                                        |
| - `p1`: Probability of success (i.e., the Bernoulli variable equals 1) for the first   |
|   Bernoulli variable `X1`.                                                             |
| - `p2`: Probability of success for the second Bernoulli variable `X2`.                 |
|                                                                                        |
| These probabilities determine how likely it is for each variable to trigger a success  |
| during the random walk. Both `p1` and `p2` are values between 0 and 1, where:          |
| - A higher value (e.g., 0.8) means a higher probability of success, leading to         |
|   potentially shorter random walks.                                                    |
| - A lower value (e.g., 0.2) means a lower probability of success, leading to           |
|   potentially longer random walks.                                                     |
|                                                                                        |
| ### `rho`                                                                              |
|                                                                                        |
| - `rho`: The correlation coefficient between `X1` and `X2`.                            |
|                                                                                        |
| This parameter determines the degree of correlation between the two Bernoulli          |
| variables:                                                                             |
| - `rho = 1.0`: Perfect positive correlation, where `X1` and `X2` are highly likely     |
|   to have the same outcome.                                                            |
| - `rho = 0.0`: No correlation, where `X1` and `X2` behave independently.               |
| - `rho = -1.0`: Perfect negative correlation, where `X1` and `X2` are likely to have   |
|   opposite outcomes.                                                                   |
|                                                                                        |
| ### Example Usage                                                                      |
|                                                                                        |
| To run a simulation with:                                                              |
| - `p1 = 0.8`                                                                           |
| - `p2 = 0.8`                                                                           |
| - `rho = 0.0`                                                                          |
|                                                                                        |
| You can set the parameters in the script like this:                                    |
|                                                                                        |
| ```python                                                                              |
| # Parameters                                                                           |
| N = 1000  # Number of samples                                                          |
| p1 = 0.8  # Probability of success for the first Bernoulli variable                    |
| p2 = 0.8  # Probability of success for the second Bernoulli variable                   |
| rho = 0.0 # Correlation coefficient between X1 and X2                                  |
| ```                                                                                    |
|                                                                                        |
| This configuration runs the simulation with high and equal probabilities of success    |
| for both `X1` and `X2`, with no correlation between them.                              |
|                                                                                        |
| ### Running the Simulation                                                             |
|                                                                                        |
| To run the simulation, execute the script in a Python environment:                     |
|                                                                                        |
| ```bash                                                                                |
| python geometrix.py                                                                    |
| ```                                                                                    |
|                                                                                        |
| The script will output statistical information about the counters and display a        |
| heatmap of the final counters after the random walk.                                   |
|                                                                                        |
| ### Visualization                                                                      |
|                                                                                        |
| The heatmap generated by the script shows the distribution of the final counters `c1`  |
| and `c2` after the random walk. The color intensity indicates the frequency of         |
| different counter outcomes, providing insights into how the chosen probabilities and   |
| correlation affect the random walk process.                                            |
+----------------------------------------------------------------------------------------+