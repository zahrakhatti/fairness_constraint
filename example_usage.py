"""
Example usage of Adaptive Sampling Interior-Point Method
Solves a simple stochastic optimization problem
"""

import numpy as np
from adaptive_sampling_ipm import adaptive_sampling_ipm


def example_simple_problem():
    """
    Example: Stochastic quadratic problem with constraints

    min E[F(x, ξ)] = E[(x - ξ)^T Q (x - ξ)]
    s.t. x1 + x2 = 1  (equality)
         x1 >= 0, x2 >= 0  (inequality)

    where ξ ~ N(μ, Σ) is random
    """

    print("Example: Stochastic Quadratic Problem")
    print("=" * 60)

    # Problem dimensions
    n = 2  # Primal variables
    p = 1  # Equality constraints
    q = 2  # Inequality constraints

    # Random target distribution
    xi_mean = np.array([2.0, 3.0])
    xi_cov = np.eye(2) * 0.5

    Q = np.array([[2.0, 0.5],
                  [0.5, 1.0]])

    # Objective function F(x, xi)
    def F_func(x, xi):
        diff = x - xi
        return 0.5 * diff.T @ Q @ diff

    # Gradient of F(x, xi)
    def grad_F_func(x, xi):
        return Q @ (x - xi)

    # Equality constraint: x1 + x2 = 1
    def c_E_func(x):
        return np.array([x[0] + x[1] - 1.0])

    # Jacobian of equality constraint
    def jac_c_E_func(x):
        return np.array([[1.0, 1.0]])

    # Inequality constraints: x >= 0 (i.e., c_I(x) = x >= 0)
    def c_I_func(x):
        return x

    # Jacobian of inequality constraints
    def jac_c_I_func(x):
        return np.eye(2)

    # Hessian of Lagrangian
    def hess_lag_func(x, lambda_E, lambda_I):
        # For quadratic problem, Hessian is constant Q
        return Q

    # Sample function: draws N samples from xi distribution
    def sample_func(N):
        return [np.random.multivariate_normal(xi_mean, xi_cov) for _ in range(N)]

    # Initial point (feasible)
    x0 = np.array([0.6, 0.4])
    lambda_E0 = np.array([0.0])
    lambda_I0 = np.array([-0.1, -0.1])

    # Sample size sequence (increasing)
    N_sequence = [10, 20, 50, 100, 200]

    # Adaptive termination parameters
    gamma_sequence = [0.5, 0.4, 0.3, 0.2, 0.1]
    epsilon_sequence = [0.1, 0.05, 0.01, 0.005, 0.001]

    # Run algorithm
    result = adaptive_sampling_ipm(
        F_func=F_func,
        grad_F_func=grad_F_func,
        c_E_func=c_E_func,
        jac_c_E_func=jac_c_E_func,
        c_I_func=c_I_func,
        jac_c_I_func=jac_c_I_func,
        hess_lag_func=hess_lag_func,
        sample_func=sample_func,
        x0=x0,
        lambda_E0=lambda_E0,
        lambda_I0=lambda_I0,
        N_sequence=N_sequence,
        gamma_sequence=gamma_sequence,
        epsilon_sequence=epsilon_sequence,
        sigma=0.2,
        epsilon_mu=0.1,
        eta1=0.95,
        eta2=0.001,
        tau=0.5,
        theta=0.1,
        delta_pi=0.01,
        omega=1.0,
        mu_0=1.0,
        pi_0=1.0,
        max_outer_iter=len(N_sequence),
        max_middle_iter=20,
        max_inner_iter=50,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Solution x: {result['x']}")
    print(f"Slack s: {result['s']}")
    print(f"Lambda_E: {result['lambda_E']}")
    print(f"Lambda_I: {result['lambda_I']}")
    print(f"\nFinal KKT error: {result['history']['kkt_error'][-1]:.3e}")
    print(f"Final feasibility: {result['history']['feasibility'][-1]:.3e}")
    print(f"Final objective: {result['history']['objective'][-1]:.3e}")

    # Check solution
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Equality constraint (should be ~1): {result['x'][0] + result['x'][1]:.6f}")
    print(f"Inequality constraint x1 (should be >= 0): {result['x'][0]:.6f}")
    print(f"Inequality constraint x2 (should be >= 0): {result['x'][1]:.6f}")

    return result


def example_rosenbrock_with_constraints():
    """
    Example: Stochastic Rosenbrock with linear constraints

    min E[F(x, ξ)] = E[(1-x1)^2 + 100(x2-x1^2)^2 + ξ1*x1 + ξ2*x2]
    s.t. x1 + 2*x2 = 2  (equality)
         x1 >= 0, x2 >= 0  (inequality)
    """

    print("\n\nExample: Stochastic Rosenbrock Problem")
    print("=" * 60)

    # Problem dimensions
    n = 2
    p = 1
    q = 2

    # Random noise distribution
    xi_mean = np.array([0.0, 0.0])
    xi_cov = np.eye(2) * 0.1

    # Objective function
    def F_func(x, xi):
        rosenbrock = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        linear = xi[0] * x[0] + xi[1] * x[1]
        return rosenbrock + linear

    # Gradient
    def grad_F_func(x, xi):
        g = np.zeros(2)
        g[0] = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2) + xi[0]
        g[1] = 200*(x[1] - x[0]**2) + xi[1]
        return g

    # Equality constraint: x1 + 2*x2 = 2
    def c_E_func(x):
        return np.array([x[0] + 2*x[1] - 2.0])

    def jac_c_E_func(x):
        return np.array([[1.0, 2.0]])

    # Inequality: x >= 0
    def c_I_func(x):
        return x

    def jac_c_I_func(x):
        return np.eye(2)

    # Hessian (approximate with identity for simplicity)
    def hess_lag_func(x, lambda_E, lambda_I):
        H = np.zeros((2, 2))
        H[0, 0] = 2 + 1200*x[0]**2 - 400*x[1]
        H[0, 1] = -400*x[0]
        H[1, 0] = -400*x[0]
        H[1, 1] = 200
        # Add regularization for positive definiteness
        return H + 0.1 * np.eye(2)

    def sample_func(N):
        return [np.random.multivariate_normal(xi_mean, xi_cov) for _ in range(N)]

    # Initial feasible point
    x0 = np.array([1.0, 0.5])
    lambda_E0 = np.array([0.0])
    lambda_I0 = np.array([-0.1, -0.1])

    N_sequence = [20, 50, 100]
    gamma_sequence = [0.5, 0.3, 0.1]
    epsilon_sequence = [0.1, 0.01, 0.001]

    result = adaptive_sampling_ipm(
        F_func=F_func,
        grad_F_func=grad_F_func,
        c_E_func=c_E_func,
        jac_c_E_func=jac_c_E_func,
        c_I_func=c_I_func,
        jac_c_I_func=jac_c_I_func,
        hess_lag_func=hess_lag_func,
        sample_func=sample_func,
        x0=x0,
        lambda_E0=lambda_E0,
        lambda_I0=lambda_I0,
        N_sequence=N_sequence,
        gamma_sequence=gamma_sequence,
        epsilon_sequence=epsilon_sequence,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Solution x: {result['x']}")
    print(f"Equality constraint value: {result['x'][0] + 2*result['x'][1]:.6f} (should be 2.0)")

    return result


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run simple example
    result1 = example_simple_problem()

    # Uncomment to run Rosenbrock example
    # result2 = example_rosenbrock_with_constraints()
