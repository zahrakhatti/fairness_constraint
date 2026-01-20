"""
Retrospective Approximation with Interior-Point Method (RA-IPM)
Implementation of Algorithm from the paper
"""

import numpy as np
from scipy.linalg import lstsq, solve
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def compute_kkt_error_sampled(x, s, lambda_E, lambda_I, grad_f_S, jac_c_E, jac_c_I, c_E, c_I):
    """
    Compute the KKT error for the subsampled problem (equation 13)
    """
    S = np.diag(s)

    stationarity = grad_f_S + jac_c_E.T @ lambda_E + jac_c_I.T @ lambda_I
    eq_feasibility = c_E
    ineq_feasibility = c_I - s
    complementarity = S @ lambda_I

    kkt_error = np.concatenate([
        stationarity,
        eq_feasibility,
        ineq_feasibility,
        complementarity
    ])

    return kkt_error


def compute_kkt_error_true(x, s, lambda_E, lambda_I, grad_f, jac_c_E, jac_c_I, c_E, c_I):
    """
    Compute the KKT error for the true problem (equation 5)
    """
    S = np.diag(s)

    stationarity = grad_f + jac_c_E.T @ lambda_E + jac_c_I.T @ lambda_I
    eq_feasibility = c_E
    ineq_feasibility = c_I - s
    complementarity = S @ lambda_I

    kkt_error = np.concatenate([
        stationarity,
        eq_feasibility,
        ineq_feasibility,
        complementarity
    ])

    return kkt_error


def check_infeasible_stationary(x, s, jac_c_E, jac_c_I, c_E, c_I, tol=1e-6):
    """
    Check if point satisfies infeasible stationarity condition (equation 9)
    """
    S = np.diag(s)

    # Condition: ∇c_E(x)^T c_E(x) + ∇c_I(x)^T (c_I(x) - s) = 0
    stationarity = jac_c_E.T @ c_E + jac_c_I.T @ (c_I - s)

    # Condition: -S(c_I(x) - s) = 0
    complementarity = -S @ (c_I - s)

    # Condition: ||c(z)|| > 0
    constraint_norm = np.linalg.norm(np.concatenate([c_E, c_I - s]))

    is_stationary = (np.linalg.norm(stationarity) < tol and
                    np.linalg.norm(complementarity) < tol and
                    constraint_norm > tol)

    return is_stationary


def compute_scaled_gradient(grad_f_S, s, mu):
    """
    Compute scaled gradient γ(z; μ) (equation 16)
    """
    return np.concatenate([grad_f_S, -mu * np.ones(len(s))])


def compute_scaled_jacobian(jac_c_E, jac_c_I, s):
    """
    Compute scaled constraint Jacobian A(z) (equation 16)
    """
    n = jac_c_E.shape[0]
    p = jac_c_E.shape[1]
    q = len(s)

    S = np.diag(s)

    A = np.zeros((p + q, n + q))
    A[:p, :n] = jac_c_E.T
    A[p:, :n] = jac_c_I.T
    A[p:, n:] = -S

    return A


def compute_scaled_hessian(hess_lag_xx, s, lambda_I, mu, use_primal=True):
    """
    Compute scaled Hessian W(z, λ; μ) (equation 17)
    Σ = μI for primal, Σ = Λ_I S for primal-dual
    """
    n = hess_lag_xx.shape[0]
    q = len(s)

    if use_primal:
        Sigma = mu * np.eye(q)
    else:
        Sigma = np.diag(lambda_I) @ np.diag(s)

    W = np.zeros((n + q, n + q))
    W[:n, :n] = hess_lag_xx
    W[n:, n:] = Sigma

    return W


def solve_normal_step(c, A, omega):
    """
    Solve normal step trust region problem (equation 18)
    min_v  1/2 ||c + Av||^2
    s.t.   ||v|| <= ω ||A^T c||
    """
    n = A.shape[1]
    delta = omega * np.linalg.norm(A.T @ c)

    # Solve trust region subproblem using dogleg or exact method
    # For simplicity, use Cauchy point approach
    g = A.T @ (A @ np.zeros(n) + c)

    if np.linalg.norm(g) < 1e-10:
        return np.zeros(n)

    # Steepest descent direction
    d = -g

    # Compute optimal step in steepest descent direction
    Ad = A @ d
    alpha_sd = (g.T @ g) / (Ad.T @ Ad)
    v_sd = alpha_sd * d

    # If within trust region, return it
    if np.linalg.norm(v_sd) <= delta:
        return v_sd
    else:
        # Project to boundary
        return (delta / np.linalg.norm(d)) * d


def solve_tangential_step(W, A, gamma, lambda_dual, v):
    """
    Solve tangential step system (equation 20)
    [W   A^T] [d    ]   [-γ - A^T λ]
    [A   0  ] [δ    ] = [-Av       ]
    """
    n = A.shape[1]
    p = A.shape[0]

    # Build KKT system
    KKT = np.zeros((n + p, n + p))
    KKT[:n, :n] = W
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A

    rhs = np.concatenate([
        -(gamma + A.T @ lambda_dual),
        -A @ v
    ])

    # Solve system (may use iterative solver for large problems)
    try:
        sol = solve(KKT, rhs, assume_a='sym')
    except:
        # Use least squares if singular
        sol, _, _, _ = lstsq(KKT, rhs)

    d = sol[:n]
    delta = sol[n:]

    return d, delta


def compute_merit_function(x, s, f_S, c_E, c_I, mu, pi):
    """
    Compute ℓ1 merit function (equation 21)
    φ(z; μ, π) = f_S(x) - μ Σ ln(s_i) + π ||c(z)||
    """
    barrier_term = -mu * np.sum(np.log(s))
    c = np.concatenate([c_E, c_I - s])
    penalty_term = pi * np.linalg.norm(c, 1)

    return f_S + barrier_term + penalty_term


def compute_model_reduction(gamma, d, A, c, pi):
    """
    Compute predicted reduction in merit model (equation 24)
    """
    linear_term = -gamma.T @ d
    constraint_improvement = pi * (np.linalg.norm(c, 1) - np.linalg.norm(c + A @ d, 1))

    return linear_term + constraint_improvement


def update_penalty_parameter(gamma, d, u, W, A, c, v, pi_prev, theta, tau, delta_pi):
    """
    Update penalty parameter π (equations 26-27)
    """
    # Compute model reduction with current penalty
    delta_m = compute_model_reduction(gamma, d, A, c, pi_prev)

    # Compute required reduction
    u_W_u = u.T @ W @ u
    min_reduction = max(0.5 * u_W_u, theta * np.linalg.norm(u)**2)

    constraint_reduction = np.linalg.norm(c, 1) - np.linalg.norm(c + A @ v, 1)
    required_reduction = min_reduction + tau * pi_prev * constraint_reduction

    # Check reduction condition
    if delta_m >= required_reduction:
        return pi_prev
    else:
        # Compute trial penalty parameter
        if constraint_reduction > 1e-10:
            pi_trial = (gamma.T @ d + min_reduction) / ((1 - tau) * constraint_reduction)
        else:
            pi_trial = pi_prev * 10  # Increase if no constraint improvement

        return pi_trial + delta_pi


def compute_max_step_fraction_to_boundary(s, d_s, eta1):
    """
    Compute maximum step size satisfying fraction-to-boundary rule (equation 28)
    s + α S d_s >= (1 - η1) s
    """
    alpha_max = 1.0

    for i in range(len(s)):
        if d_s[i] < 0:
            alpha_i = -eta1 / d_s[i]
            alpha_max = min(alpha_max, alpha_i)

    return alpha_max


def armijo_line_search(x, s, d_x, d_s, f_S_func, c_E_func, c_I_func,
                       mu, pi, gamma, d, A, c, eta2, alpha_max):
    """
    Backtracking line search with Armijo condition (equation 30)
    """
    phi_0 = compute_merit_function(x, s, f_S_func(x), c_E_func(x), c_I_func(x), mu, pi)
    delta_m = compute_model_reduction(gamma, d, A, c, pi)

    alpha = alpha_max
    max_iter = 50

    for i in range(max_iter):
        x_new = x + alpha * d_x
        s_new = s + alpha * np.diag(s) @ d_s  # Scaled step

        # Check if s_new is positive
        if np.any(s_new <= 0):
            alpha *= 0.5
            continue

        phi_new = compute_merit_function(x_new, s_new, f_S_func(x_new),
                                        c_E_func(x_new), c_I_func(x_new), mu, pi)

        # Armijo condition
        if phi_new <= phi_0 - eta2 * alpha * delta_m:
            return alpha

        alpha *= 0.5

    return alpha


def update_dual_multiplier(gamma, A, lambda_curr, delta):
    """
    Update dual multiplier (equations 32-33)
    β = argmin_β ||γ + A^T(λ + β δ)||
    """
    # This is a 1D optimization problem
    # The optimal β is found by differentiating wrt β

    ATdelta = A.T @ delta
    residual_0 = gamma + A.T @ lambda_curr

    # β* = -(residual_0^T A^T δ) / (A^T δ)^T (A^T δ)
    numerator = -residual_0.T @ ATdelta
    denominator = ATdelta.T @ ATdelta

    if denominator > 1e-10:
        beta = numerator / denominator
    else:
        beta = 0.0

    # Clip to [0, 1]
    beta = np.clip(beta, 0.0, 1.0)

    return beta


def check_barrier_termination(x, s, lambda_E, lambda_I, grad_f_S, jac_c_E,
                               jac_c_I, c_E, c_I, mu, epsilon_mu):
    """
    Check barrier subproblem termination conditions (equation 15)
    """
    S = np.diag(s)

    # Dual infeasibility
    dual_inf = np.linalg.norm(grad_f_S + jac_c_E.T @ lambda_E + jac_c_I.T @ lambda_I, np.inf)

    # Complementarity
    comp_inf = np.linalg.norm(S @ lambda_I + mu * np.ones(len(s)), np.inf)

    # Primal feasibility
    c = np.concatenate([c_E, c_I - s])
    primal_inf = np.linalg.norm(c, np.inf)

    tol = epsilon_mu * mu

    return (dual_inf <= tol and comp_inf <= tol and primal_inf <= tol)


def adaptive_sampling_ipm(
    # Problem functions
    F_func,  # F(x, xi) - objective for single sample
    grad_F_func,  # Gradient of F(x, xi)
    c_E_func,  # Equality constraints
    jac_c_E_func,  # Jacobian of equality constraints
    c_I_func,  # Inequality constraints
    jac_c_I_func,  # Jacobian of inequality constraints
    hess_lag_func,  # Hessian of Lagrangian
    sample_func,  # Function to draw samples
    # Initial point
    x0,
    lambda_E0,
    lambda_I0,
    # Algorithm parameters
    N_sequence,  # Sequence of sample sizes
    gamma_sequence,  # Adaptive termination parameters γ_k
    epsilon_sequence,  # Adaptive termination parameters ε_k
    sigma=0.1,  # Barrier reduction factor
    epsilon_mu=0.1,  # Barrier tolerance
    eta1=0.99,  # Fraction-to-boundary parameter
    eta2=0.01,  # Armijo parameter
    tau=0.5,  # Penalty parameter
    theta=0.1,  # Curvature parameter
    delta_pi=0.1,  # Penalty increment
    omega=1.0,  # Normal step parameter
    mu_0=1.0,  # Initial barrier parameter
    pi_0=1.0,  # Initial penalty parameter
    max_outer_iter=100,
    max_middle_iter=50,
    max_inner_iter=100,
    verbose=True
):
    """
    Main Algorithm: Retrospective Approximation with Interior-Point Method

    Algorithm 1 from the paper
    """

    # Initialize
    n = len(x0)
    q = len(lambda_I0)
    p = len(lambda_E0)

    # Initialize primal variables
    x_k = x0.copy()
    c_I_init = c_I_func(x_k)
    s_k = np.maximum(c_I_init, 0.1 * np.ones(q))  # Ensure s > 0

    # Initialize dual variables
    lambda_E_k = lambda_E0.copy()
    lambda_I_k = lambda_I0.copy()

    # Ensure lambda_I < 0
    lambda_I_k = np.minimum(lambda_I_k, -0.1 * np.ones(q))

    history = {
        'kkt_error': [],
        'feasibility': [],
        'objective': []
    }

    # Outer loop
    for k in range(len(N_sequence)):
        N_k = N_sequence[k]
        gamma_k = gamma_sequence[k] if k < len(gamma_sequence) else gamma_sequence[-1]
        epsilon_k = epsilon_sequence[k] if k < len(epsilon_sequence) else epsilon_sequence[-1]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Outer iteration k={k}, Sample size N_k={N_k}")
            print(f"{'='*60}")

        # Draw sample set S_k
        S_k = sample_func(N_k)

        # Define subsampled objective
        def f_S_k(x):
            return np.mean([F_func(x, xi) for xi in S_k])

        def grad_f_S_k(x):
            return np.mean([grad_F_func(x, xi) for xi in S_k], axis=0)

        # Check infeasibility
        c_E_k = c_E_func(x_k)
        c_I_k = c_I_func(x_k)
        jac_c_E_k = jac_c_E_func(x_k)
        jac_c_I_k = jac_c_I_func(x_k)

        if check_infeasible_stationary(x_k, s_k, jac_c_E_k, jac_c_I_k, c_E_k, c_I_k):
            if verbose:
                print("Detected infeasible stationary point. Terminating.")
            break

        # Initialize middle loop
        x_kj = x_k.copy()
        s_kj = s_k.copy()
        lambda_E_kj = lambda_E_k.copy()
        lambda_I_kj = lambda_I_k.copy()

        mu_kj = mu_0 if k == 0 else sigma * mu_k_prev

        # Compute initial KKT error for adaptive termination
        grad_f_S_k0 = grad_f_S_k(x_k)
        kkt_S_k0 = compute_kkt_error_sampled(x_k, s_k, lambda_E_k, lambda_I_k,
                                             grad_f_S_k0, jac_c_E_k, jac_c_I_k,
                                             c_E_k, c_I_k)
        kkt_S_k0_norm = np.linalg.norm(kkt_S_k0)

        # Middle loop (barrier parameter reduction)
        for j in range(max_middle_iter):
            if verbose:
                print(f"\n  Middle iteration j={j}, μ={mu_kj:.2e}")

            # Check adaptive termination
            grad_f_S_kj = grad_f_S_k(x_kj)
            c_E_kj = c_E_func(x_kj)
            c_I_kj = c_I_func(x_kj)
            jac_c_E_kj = jac_c_E_func(x_kj)
            jac_c_I_kj = jac_c_I_func(x_kj)

            kkt_S_kj = compute_kkt_error_sampled(x_kj, s_kj, lambda_E_kj, lambda_I_kj,
                                                 grad_f_S_kj, jac_c_E_kj, jac_c_I_kj,
                                                 c_E_kj, c_I_kj)
            kkt_S_kj_norm = np.linalg.norm(kkt_S_kj)

            if kkt_S_kj_norm <= gamma_k * kkt_S_k0_norm + epsilon_k:
                if verbose:
                    print(f"  Adaptive termination satisfied at j={j}")
                break

            if check_infeasible_stationary(x_kj, s_kj, jac_c_E_kj, jac_c_I_kj, c_E_kj, c_I_kj):
                if verbose:
                    print("  Detected infeasible stationary point in middle loop.")
                break

            # Initialize inner loop
            x_kjl = x_kj.copy()
            s_kjl = s_kj.copy()
            lambda_E_kjl = lambda_E_kj.copy()
            lambda_I_kjl = lambda_I_kj.copy()

            pi_kjl = pi_0

            # Inner loop (interior-point iterations)
            for ell in range(max_inner_iter):
                # Evaluate functions
                f_S_kjl = f_S_k(x_kjl)
                grad_f_S_kjl = grad_f_S_k(x_kjl)
                c_E_kjl = c_E_func(x_kjl)
                c_I_kjl = c_I_func(x_kjl)
                jac_c_E_kjl = jac_c_E_func(x_kjl)
                jac_c_I_kjl = jac_c_I_func(x_kjl)
                hess_lag_kjl = hess_lag_func(x_kjl, lambda_E_kjl, lambda_I_kjl)

                # Check barrier termination
                if check_barrier_termination(x_kjl, s_kjl, lambda_E_kjl, lambda_I_kjl,
                                            grad_f_S_kjl, jac_c_E_kjl, jac_c_I_kjl,
                                            c_E_kjl, c_I_kjl, mu_kj, epsilon_mu):
                    if verbose:
                        print(f"    Barrier termination at ℓ={ell}")
                    break

                if check_infeasible_stationary(x_kjl, s_kjl, jac_c_E_kjl, jac_c_I_kjl,
                                               c_E_kjl, c_I_kjl):
                    if verbose:
                        print(f"    Infeasible stationary at ℓ={ell}")
                    break

                # Compute scaled quantities
                gamma_kjl = compute_scaled_gradient(grad_f_S_kjl, s_kjl, mu_kj)
                A_kjl = compute_scaled_jacobian(jac_c_E_kjl, jac_c_I_kjl, s_kjl)
                W_kjl = compute_scaled_hessian(hess_lag_kjl, s_kjl, lambda_I_kjl, mu_kj)

                c_kjl = np.concatenate([c_E_kjl, c_I_kjl - s_kjl])

                # Compute normal step
                v_kjl = solve_normal_step(c_kjl, A_kjl, omega)

                # Compute tangential step
                lambda_kjl = np.concatenate([lambda_E_kjl, lambda_I_kjl])
                d_kjl, delta_kjl = solve_tangential_step(W_kjl, A_kjl, gamma_kjl,
                                                         lambda_kjl, v_kjl)

                u_kjl = d_kjl - v_kjl

                # Update penalty parameter
                pi_kjl = update_penalty_parameter(gamma_kjl, d_kjl, u_kjl, W_kjl,
                                                 A_kjl, c_kjl, v_kjl, pi_kjl,
                                                 theta, tau, delta_pi)

                # Extract primal components
                d_x = d_kjl[:n]
                d_s = d_kjl[n:]

                # Compute maximum step size
                alpha_max = compute_max_step_fraction_to_boundary(s_kjl, d_s, eta1)

                # Line search
                alpha_kjl = armijo_line_search(x_kjl, s_kjl, d_x, d_s,
                                              f_S_k, c_E_func, c_I_func,
                                              mu_kj, pi_kjl, gamma_kjl, d_kjl,
                                              A_kjl, c_kjl, eta2, alpha_max)

                # Update primal variables
                x_kjl = x_kjl + alpha_kjl * d_x
                s_kjl = s_kjl + alpha_kjl * np.diag(s_kjl) @ d_s

                # Update dual variables
                beta_kjl = update_dual_multiplier(gamma_kjl, A_kjl, lambda_kjl, delta_kjl)
                lambda_kjl = lambda_kjl + beta_kjl * delta_kjl

                lambda_E_kjl = lambda_kjl[:p]
                lambda_I_kjl = lambda_kjl[p:]

                # Slack reset
                s_kjl = np.maximum(s_kjl, c_I_func(x_kjl))

                if verbose and ell % 10 == 0:
                    print(f"    Inner ℓ={ell}, α={alpha_kjl:.3e}, ||c||={np.linalg.norm(c_kjl):.3e}")

            # Update for next barrier iteration
            x_kj = x_kjl.copy()
            s_kj = s_kjl.copy()
            lambda_E_kj = lambda_E_kjl.copy()
            lambda_I_kj = lambda_I_kjl.copy()

            # Reduce barrier parameter
            mu_kj = sigma * mu_kj

        # Update for next outer iteration
        x_k = x_kj.copy()
        s_k = s_kj.copy()
        lambda_E_k = lambda_E_kj.copy()
        lambda_I_k = lambda_I_kj.copy()

        mu_k_prev = mu_kj

        # Compute true KKT error (if true gradient available)
        # For now, use subsampled version
        c_E_k = c_E_func(x_k)
        c_I_k = c_I_func(x_k)
        jac_c_E_k = jac_c_E_func(x_k)
        jac_c_I_k = jac_c_I_func(x_k)
        grad_f_S_k_final = grad_f_S_k(x_k)

        kkt_error = compute_kkt_error_sampled(x_k, s_k, lambda_E_k, lambda_I_k,
                                             grad_f_S_k_final, jac_c_E_k, jac_c_I_k,
                                             c_E_k, c_I_k)

        history['kkt_error'].append(np.linalg.norm(kkt_error))
        history['feasibility'].append(np.linalg.norm(np.concatenate([c_E_k, c_I_k - s_k])))
        history['objective'].append(f_S_k(x_k))

        if verbose:
            print(f"\nOuter iteration k={k} complete:")
            print(f"  KKT error: {history['kkt_error'][-1]:.3e}")
            print(f"  Feasibility: {history['feasibility'][-1]:.3e}")
            print(f"  Objective: {history['objective'][-1]:.3e}")

        # Check convergence
        if history['kkt_error'][-1] < 1e-6 and history['feasibility'][-1] < 1e-6:
            if verbose:
                print("\nConverged!")
            break

    return {
        'x': x_k,
        's': s_k,
        'lambda_E': lambda_E_k,
        'lambda_I': lambda_I_k,
        'history': history
    }


# Example usage
if __name__ == "__main__":
    print("Adaptive Sampling Interior-Point Method Implementation")
    print("=" * 60)
    print("\nThis is a framework implementation.")
    print("To use, provide:")
    print("  - F_func: objective for single sample")
    print("  - Constraint functions c_E, c_I")
    print("  - Their Jacobians and Hessian")
    print("  - Sample function")
    print("  - Initial point and parameters")
