# Adaptive Sampling Interior-Point Method Implementation

This repository contains a Python implementation of the **Retrospective Approximation with Interior-Point Method (RA-IPM)** algorithm described in the paper "Adaptive Sampling".

## Overview

The algorithm solves stochastic nonlinear optimization problems of the form:

```
min  E[F(x,ξ)]
s.t. c_E(x) = 0    (equality constraints)
     c_I(x) >= 0   (inequality constraints)
```

where `E[F(x,ξ)]` is the expected value of the objective function over random variable `ξ`.

## Algorithm Structure

The implementation follows a **three-level nested loop structure**:

1. **Outer Loop** (index k): Increases sample sizes `N_k` progressively
2. **Middle Loop** (index j): Decreases barrier parameters `μ_j` geometrically
3. **Inner Loop** (index ℓ): Performs interior-point iterations for each barrier subproblem

### Key Features

- **Adaptive Sampling**: Solves subsampled problems with increasing sample sizes
- **Interior-Point Method**: Uses logarithmic barrier for inequality constraints
- **Normal-Tangential Decomposition**: Separates feasibility and optimality improvements
- **Merit Function with Line Search**: Ensures global convergence
- **Fraction-to-Boundary Rule**: Maintains strict positivity of slack variables
- **Warm-Starting**: Carries over solutions between outer iterations

## Files

- `adaptive_sampling_ipm.py`: Main algorithm implementation
- `example_usage.py`: Example problems demonstrating usage
- `README_IPM.md`: This documentation file

## Usage

### Basic Structure

```python
from adaptive_sampling_ipm import adaptive_sampling_ipm

# Define problem functions
def F_func(x, xi):
    """Objective function for single sample"""
    pass

def grad_F_func(x, xi):
    """Gradient of objective"""
    pass

def c_E_func(x):
    """Equality constraints"""
    pass

def jac_c_E_func(x):
    """Jacobian of equality constraints"""
    pass

def c_I_func(x):
    """Inequality constraints (c_I(x) >= 0)"""
    pass

def jac_c_I_func(x):
    """Jacobian of inequality constraints"""
    pass

def hess_lag_func(x, lambda_E, lambda_I):
    """Hessian of Lagrangian"""
    pass

def sample_func(N):
    """Draw N samples from distribution of xi"""
    pass

# Set initial point
x0 = ...
lambda_E0 = ...
lambda_I0 = ...

# Define sample size sequence
N_sequence = [10, 20, 50, 100, 200]

# Define adaptive termination parameters
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
    epsilon_sequence=epsilon_sequence
)

# Access results
x_optimal = result['x']
kkt_error = result['history']['kkt_error']
```

### Running Examples

```bash
python example_usage.py
```

The example file contains:
1. Simple stochastic quadratic problem with linear constraints
2. Stochastic Rosenbrock problem with constraints

## Algorithm Parameters

### Required Parameters

- `N_sequence`: List of increasing sample sizes for outer iterations
- `gamma_sequence`: Adaptive termination parameters γ_k ∈ (0,1)
- `epsilon_sequence`: Adaptive termination parameters ε_k >= 0

### Optional Parameters (with defaults)

- `sigma=0.1`: Barrier reduction factor (μ_{j+1} = σ μ_j)
- `epsilon_mu=0.1`: Barrier subproblem tolerance
- `eta1=0.99`: Fraction-to-boundary parameter (keeps s > 0)
- `eta2=0.01`: Armijo line search parameter
- `tau=0.5`: Penalty parameter update threshold
- `theta=0.1`: Curvature parameter for tangential step
- `delta_pi=0.1`: Penalty parameter increment
- `omega=1.0`: Normal step trust region parameter
- `mu_0=1.0`: Initial barrier parameter
- `pi_0=1.0`: Initial penalty parameter

## Output

The algorithm returns a dictionary with:

- `x`: Final primal solution
- `s`: Final slack variables
- `lambda_E`: Final equality constraint multipliers
- `lambda_I`: Final inequality constraint multipliers
- `history`: Dictionary containing:
  - `kkt_error`: KKT error norm at each outer iteration
  - `feasibility`: Constraint violation at each outer iteration
  - `objective`: Objective value at each outer iteration

## Mathematical Components

### KKT Error (Equation 5)

The KKT error measures progress toward first-order optimality:

```
T(z,λ) = [∇f(x) + ∇c_E(x)λ_E + ∇c_I(x)λ_I]
         [c_E(x)                            ]
         [c_I(x) - s                        ]
         [S λ_I                             ]
```

### Normal Step (Equation 18)

Solves trust region subproblem for feasibility:

```
min_v  1/2 ||c + Av||²
s.t.   ||v|| <= ω ||A^T c||
```

### Tangential Step (Equation 20)

Solves perturbed Newton system:

```
[W   A^T] [d]   [-γ - A^T λ]
[A   0  ] [δ] = [-Av       ]
```

### Merit Function (Equation 21)

ℓ1 penalty function for line search:

```
φ(z; μ, π) = f_S(x) - μ Σ ln(s_i) + π ||c(z)||
```

## Convergence Guarantees

Under appropriate assumptions (see Theorem 3.1 in paper):

- If sample sizes N_k → ∞
- And termination parameters satisfy 0 <= γ_k <= γ < 1, ε_k → 0
- Then ||T(z_k, λ_k)|| → 0
- And limit points satisfy KKT conditions of true problem

## Implementation Notes

1. **Slack Variables**: Maintained strictly positive via fraction-to-boundary rule
2. **Dual Variables**: For inequality constraints, λ_I < 0 is enforced
3. **Slack Reset**: After each iterate update, s is reset to ensure c_I(x) - s <= 0
4. **System Solving**: Uses direct solvers (scipy.linalg.solve) with least squares fallback

## Dependencies

```python
numpy
scipy
```

## Example: Simple Quadratic Problem

```python
# Problem: min E[(x - ξ)^T Q (x - ξ)]
#          s.t. x1 + x2 = 1
#               x1, x2 >= 0

Q = np.array([[2.0, 0.5], [0.5, 1.0]])

def F_func(x, xi):
    diff = x - xi
    return 0.5 * diff.T @ Q @ diff

def grad_F_func(x, xi):
    return Q @ (x - xi)

# ... (see example_usage.py for complete code)
```

## References

Based on the paper "Adaptive Sampling" which describes:
- Retrospective approximation framework
- Interior-point methods with barrier functions
- Normal-tangential step decomposition
- Global convergence theory

## License

This implementation is for educational and research purposes.

## Contact

For questions or issues, please refer to the paper or contact the authors.
