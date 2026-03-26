import torch
from torch.optim.optimizer import Optimizer, required
import time
from config import get_args

args = get_args()


class StochasticSQP(Optimizer):
    """Stochastic Sequential Quadratic Programming optimizer for constrained optimization."""

    # Class constants
    SIGMA = 0.5
    ETA = 1e-4
    BUFFER = 0
    EPS = 1e-6
    THRESH = 1e12  # Threshold for "extreme" bounds

    def __init__(self, params, lr=required, epsilon=1e-5,
                 upper_bound=[1e-5, 1e-5, 1e-5, 1e-5],
                 lower_bound=[1e-5, 1e-5, 1e-5, 1e-5],
                 n_parameters=0, n_constrs=0,
                 merit_param_init=1, ratio_param_init=1,
                 step_size_decay=0.5, step_opt=1,
                 problem=None, mu=100, beta2=0.999):
        defaults = dict()
        super(StochasticSQP, self).__init__(params, defaults)

        self.epsilon = epsilon
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.n_parameters = n_parameters
        self.n_constrs = n_constrs
        self.merit_param = merit_param_init
        self.ratio_param = ratio_param_init
        self.step_size = lr
        self.step_size_init = lr
        self.step_size_decay = step_size_decay
        self.trial_merit = 1.0
        self.trial_ratio = 1.0
        self.norm_d = 0.0
        self.initial_params = params
        self.step_opt = step_opt
        self.problem = problem
        self.mu = mu
        self.beta2 = beta2

        # Timing statistics
        self.timing_stats = {
            "matrix_construction": 0.0,
            "system_solving": 0.0,
            "constraint_checking": 0.0,
            "candidate_evaluation": 0.0,
            "step_computation": 0.0,
            "total_linalg": 0.0,
            "calls": 0
        }

    def __setstate__(self, state):
        super(StochasticSQP, self).__setstate__(state)

    def print_timing_stats(self):
        """Print timing statistics averaged over all optimizer steps."""
        if self.timing_stats["calls"] == 0:
            print("No timing data available yet.")
            return

        calls = self.timing_stats["calls"]
        print("\n=== StochasticSQP Timing Statistics ===")
        print(f"Averaged over {calls} optimizer steps:")
        print(f"Matrix construction:   {self.timing_stats['matrix_construction'] / calls:.4f}s")
        print(f"Linear system solving: {self.timing_stats['system_solving'] / calls:.4f}s")
        print(f"Constraint checking:   {self.timing_stats['constraint_checking'] / calls:.4f}s")
        print(f"Candidate evaluation:  {self.timing_stats['candidate_evaluation'] / calls:.4f}s")
        print(f"Step computation:      {self.timing_stats['step_computation'] / calls:.4f}s")
        print(f"Total linear algebra:  {self.timing_stats['total_linalg'] / calls:.4f}s")
        print("=======================================\n")

    # ============================================================================
    # Helper Methods for Linear System Solving
    # ============================================================================

    def _solve_full_system(self, ls_matrix_full, g, c_adj):
        """Solve the full KKT system with adjusted constraints."""
        solve_start = time.time()
        ls_rhs_adj = -torch.cat((g, c_adj), dim=0)
        system_solution = torch.linalg.solve(ls_matrix_full, ls_rhs_adj)
        d = system_solution[:self.n_parameters]
        y = system_solution[self.n_parameters:]
        self.timing_stats["system_solving"] += time.time() - solve_start
        return d, y

    def _solve_subset_system(self, H, J, g, c_adj_val, constraint_indices):
        """Solve KKT system with a subset of constraints."""
        solve_start = time.time()
        n_subset = len(constraint_indices)
        J_sub = J[constraint_indices]
        zeros_sub = torch.zeros(n_subset, n_subset, device=J.device)

        ls_matrix_sub = torch.cat((
            torch.cat((H, J_sub.T), dim=1),
            torch.cat((J_sub, zeros_sub), dim=1)
        ), dim=0)

        ls_rhs_sub = -torch.cat((g, c_adj_val), dim=0)
        system_solution = torch.linalg.solve(ls_matrix_sub, ls_rhs_sub)
        d = system_solution[:self.n_parameters]
        y = system_solution[self.n_parameters:self.n_parameters + n_subset]

        self.timing_stats["system_solving"] += time.time() - solve_start
        return d, y

    def _is_feasible(self, d, c, J, constraints_to_check=None):
        """Check if direction d satisfies specified constraints."""
        check_start = time.time()

        if constraints_to_check is None:
            constraints_to_check = range(self.n_constrs)

        for i in constraints_to_check:
            constr_val = c[i] + torch.matmul(J[i], d)

            if i < len(self.lower_bound) and self.lower_bound[i] < self.THRESH:
                if constr_val < -self.lower_bound[i]:
                    self.timing_stats["constraint_checking"] += time.time() - check_start
                    return False

            if i < len(self.upper_bound) and self.upper_bound[i] < self.THRESH:
                if constr_val > self.upper_bound[i]:
                    self.timing_stats["constraint_checking"] += time.time() - check_start
                    return False

        self.timing_stats["constraint_checking"] += time.time() - check_start
        return True

    def _create_full_y(self, y_sub, active_indices):
        """Create full y vector with zeros for inactive constraints."""
        y_full = torch.zeros(self.n_constrs, device=y_sub.device)
        for idx, i in enumerate(active_indices):
            y_full[i] = y_sub[idx]
        return y_full

    def _compute_objective(self, d, H, g):
        """Compute quadratic objective value."""
        eval_start = time.time()
        obj_val = torch.matmul(g, d) + 0.5 * torch.matmul(d, torch.matmul(H, d))
        self.timing_stats["candidate_evaluation"] += time.time() - eval_start
        return obj_val

    # ============================================================================
    # Constraint Case Handlers
    # ============================================================================

    def _solve_unconstrained(self, H, g):
        """Solve unconstrained problem."""
        solve_start = time.time()
        d = torch.linalg.solve(H, -g)
        self.timing_stats["system_solving"] += time.time() - solve_start
        return d, None, []

    def _solve_one_constraint(self, H, g, c, J, ls_matrix_full):
        """Solve problem with one constraint."""
        # Try unconstrained first
        d_m, _, _ = self._solve_unconstrained(H, g)

        if abs(c + torch.matmul(J, d_m)) <= self.epsilon:
            return d_m, None, []

        # Solve with constraint active at both bounds
        c_adj_upper = (c - self.epsilon).to(args.device)
        c_adj_lower = (c + self.epsilon).to(args.device)

        d_u, y_u = self._solve_full_system(ls_matrix_full, g, c_adj_upper)
        d_l, y_l = self._solve_full_system(ls_matrix_full, g, c_adj_lower)

        # Choose the one with lower objective
        obj_u = self._compute_objective(d_u, H, g)
        obj_l = self._compute_objective(d_l, H, g)

        if obj_u < obj_l:
            return d_u, y_u, [0]
        return d_l, y_l, [0]

    def _solve_two_constraints(self, H, g, c, J, ls_matrix_full):
        """Solve problem with two constraints."""
        # Try unconstrained first
        d_mm, _, _ = self._solve_unconstrained(H, g)

        if self._is_feasible(d_mm, c, J):
            return d_mm, None, []

        # Prepare constraint adjustments
        c_adj_upper = torch.zeros_like(c).to(args.device)
        c_adj_lower = torch.zeros_like(c).to(args.device)
        c_adj_upper[0] = c[0] - self.upper_bound[0]
        c_adj_lower[0] = c[0] + self.lower_bound[0]
        c_adj_upper[1] = c[1] - self.upper_bound[1]
        c_adj_lower[1] = c[1] + self.lower_bound[1]

        # Try single constraint solutions
        single_candidates = self._try_single_constraints_2d(H, g, c, J, c_adj_upper, c_adj_lower)

        if single_candidates:
            return min(single_candidates, key=lambda x: x[2])[:3]

        # Try double constraint solutions
        double_candidates = self._try_double_constraints(H, g, c, J, ls_matrix_full, c_adj_upper, c_adj_lower)

        if double_candidates:
            return min(double_candidates, key=lambda x: x[2])[:3]

        # Fallback
        self.step_size *= self.step_size_decay
        return d_mm, None, []

    def _try_single_constraints_2d(self, H, g, c, J, c_adj_upper, c_adj_lower):
        """Try solutions with one active constraint for 2D case."""
        candidates = []

        # Try each constraint at upper and lower bounds
        for i in [0, 1]:
            other_i = 1 - i

            if self.upper_bound[i] < self.THRESH:
                try:
                    d, y = self._solve_subset_system(H, J, g, c_adj_upper[i:i+1], [i])
                    if self._is_feasible(d, c, J, [other_i]):
                        y_full = self._create_full_y(y, [i])
                        candidates.append((d, y_full, self._compute_objective(d, H, g), [i]))
                except Exception:
                    pass

            if self.lower_bound[i] < self.THRESH:
                try:
                    d, y = self._solve_subset_system(H, J, g, c_adj_lower[i:i+1], [i])
                    if self._is_feasible(d, c, J, [other_i]):
                        y_full = self._create_full_y(y, [i])
                        candidates.append((d, y_full, self._compute_objective(d, H, g), [i]))
                except Exception:
                    pass

        return candidates

    def _try_double_constraints(self, H, g, c, J, ls_matrix_full, c_adj_upper, c_adj_lower):
        """Try solutions with two active constraints."""
        candidates = []

        # Try all combinations of bounds
        combinations = [
            (c_adj_upper, self.upper_bound, 'upper-upper'),
            (c_adj_lower, self.lower_bound, 'lower-lower'),
            (torch.cat((c_adj_upper[0:1], c_adj_lower[1:2])),
             [self.upper_bound[0], self.lower_bound[1]], 'upper-lower'),
            (torch.cat((c_adj_lower[0:1], c_adj_upper[1:2])),
             [self.lower_bound[0], self.upper_bound[1]], 'lower-upper'),
        ]

        for c_adj, bounds, name in combinations:
            if all(b < self.THRESH for b in bounds):
                try:
                    d, y = self._solve_full_system(ls_matrix_full, g, c_adj)
                    candidates.append((d, y, self._compute_objective(d, H, g), [0, 1]))
                except Exception:
                    pass

        return candidates

    def _solve_four_constraints(self, H, g, c, J, ls_matrix_full):
        """Solve problem with four constraints (upper bounds only)."""
        # Try unconstrained first
        d_mmmm, _, _ = self._solve_unconstrained(H, g)

        if self._is_feasible(d_mmmm, c, J):
            return d_mmmm, None, []

        # Prepare constraint adjustments
        c_adj_upper = torch.tensor([c[i] - self.upper_bound[i] for i in range(4)]).to(args.device)

        # Try single active constraints
        single_candidates = self._try_single_constraints_4d(H, g, c, J, c_adj_upper)
        if single_candidates:
            return min(single_candidates, key=lambda x: x[2])[:3]

        # Try pairs of active constraints
        double_candidates = self._try_double_constraints_4d(H, g, c, J, c_adj_upper)
        if double_candidates:
            return min(double_candidates, key=lambda x: x[2])[:3]

        # Try triples of active constraints
        triple_candidates = self._try_triple_constraints_4d(H, g, c, J, c_adj_upper)
        if triple_candidates:
            return min(triple_candidates, key=lambda x: x[2])[:3]

        # Try all four constraints
        if all(self.upper_bound[i] < self.THRESH for i in range(4)):
            try:
                d, y = self._solve_full_system(ls_matrix_full, g, c_adj_upper)
                return d, y, list(range(4))
            except Exception:
                pass

        # Fallback
        self.step_size *= self.step_size_decay
        return d_mmmm, None, []

    def _try_single_constraints_4d(self, H, g, c, J, c_adj_upper):
        """Try solutions with one active constraint for 4D case."""
        candidates = []

        for i in range(4):
            if self.upper_bound[i] < self.THRESH:
                try:
                    d, y = self._solve_subset_system(H, J, g, c_adj_upper[i:i+1], [i])
                    other_constraints = [j for j in range(4) if j != i]

                    if self._is_feasible(d, c, J, other_constraints):
                        y_full = self._create_full_y(y, [i])
                        candidates.append((d, y_full, self._compute_objective(d, H, g), [i]))
                except Exception:
                    pass

        return candidates

    def _try_double_constraints_4d(self, H, g, c, J, c_adj_upper):
        """Try solutions with two active constraints for 4D case."""
        candidates = []

        for i in range(4):
            for j in range(i + 1, 4):
                if self.upper_bound[i] < self.THRESH and self.upper_bound[j] < self.THRESH:
                    try:
                        c_adj_val = torch.cat((c_adj_upper[i:i+1], c_adj_upper[j:j+1]))
                        d, y = self._solve_subset_system(H, J, g, c_adj_val, [i, j])
                        other_constraints = [k for k in range(4) if k not in [i, j]]

                        if self._is_feasible(d, c, J, other_constraints):
                            y_full = self._create_full_y(y, [i, j])
                            candidates.append((d, y_full, self._compute_objective(d, H, g), [i, j]))
                    except Exception:
                        pass

        return candidates

    def _try_triple_constraints_4d(self, H, g, c, J, c_adj_upper):
        """Try solutions with three active constraints for 4D case."""
        candidates = []

        for excluded_i in range(4):
            active_indices = [j for j in range(4) if j != excluded_i]

            if all(self.upper_bound[j] < self.THRESH for j in active_indices):
                try:
                    c_adj_val = torch.cat([c_adj_upper[j:j+1] for j in active_indices])
                    d, y = self._solve_subset_system(H, J, g, c_adj_val, active_indices)

                    if self._is_feasible(d, c, J, [excluded_i]):
                        y_full = self._create_full_y(y, active_indices)
                        candidates.append((d, y_full, self._compute_objective(d, H, g), active_indices))
                except Exception:
                    pass

        return candidates

    # ============================================================================
    # Main Step Method
    # ============================================================================

    def step(self, closure=None):
        """Perform a single optimization step."""
        start_time_total = time.time()

        # Extract state
        J = self.state['J']
        g = self.state['g']
        c = self.state['c']
        f = self.state['f']

        # Update iteration counter and adaptive scaling
        if 'iter' not in self.state:
            self.state['iter'] = 0
            self.state['g_square_sum'] = (1 - self.beta2) * g**2
        else:
            self.state['iter'] += 1
            self.state['g_square_sum'] = (self.beta2 * self.state['g_square_sum'] +
                                          (1 - self.beta2) * g**2)

        # Construct matrices
        matrix_construction_start = time.time()
        H_diag = torch.sqrt(self.state['g_square_sum'] + self.mu)
        H = torch.diag(H_diag)
        zeros = torch.zeros(self.n_constrs, self.n_constrs, device=J.device)

        ls_matrix_full = torch.cat((
            torch.cat((H, torch.transpose(J, 0, 1)), dim=1),
            torch.cat((J, zeros), dim=1)
        ), dim=0)

        self.timing_stats["matrix_construction"] += time.time() - matrix_construction_start

        # Solve based on number of constraints
        if self.n_constrs == 0:
            d, y, active_constraints = self._solve_unconstrained(H, g)
        elif self.n_constrs == 1:
            d, y, active_constraints = self._solve_one_constraint(H, g, c, J, ls_matrix_full)
        elif self.n_constrs == 2:
            d, y, active_constraints = self._solve_two_constraints(H, g, c, J, ls_matrix_full)
        elif self.n_constrs == 4:
            d, y, active_constraints = self._solve_four_constraints(H, g, c, J, ls_matrix_full)
        else:
            raise ValueError(f"Unsupported number of constraints: {self.n_constrs}")

        # Compute step metrics
        step_comp_start = time.time()

        if torch.linalg.norm(d, ord=2) <= torch.tensor(1e-8, device=args.device):
            self.trial_merit = 10**10
            self.trial_ratio = 10**10
            self.step_size = 1
        else:
            self.norm_d = torch.norm(d)
            if y is not None:
                self.kkt_norm = torch.norm(g + torch.matmul(torch.transpose(J, 0, 1), y), float('inf'))
            else:
                self.kkt_norm = torch.norm(g, float('inf'))

            dHd = torch.matmul(torch.matmul(d, H), d)
            gd = torch.matmul(g, d)
            gd_plus_max_dHd_0 = gd + torch.max(dHd, torch.tensor(0, device=args.device))
            c_norm_1 = torch.linalg.norm(c, ord=1)

            if gd_plus_max_dHd_0 <= 0:
                self.trial_merit = 10**10
            else:
                self.trial_merit = ((1 - self.SIGMA) * c_norm_1) / gd_plus_max_dHd_0

            if self.merit_param > self.trial_merit:
                self.merit_param = self.trial_merit * (1 - self.EPS)

            delta_q = -self.merit_param * (gd + 2 * torch.max(dHd, torch.tensor(0, device=args.device))) + c_norm_1
            self.trial_ratio = delta_q / (self.merit_param * self.norm_d ** 2)

            if self.ratio_param > self.trial_ratio:
                self.ratio_param = self.trial_ratio * (1 - self.EPS)

        # Update state
        self.state['merit_param'] = self.merit_param
        self.state['cur_merit_f'] = self.merit_param * f + torch.linalg.norm(c, 1)
        self.state['phi_new'] = self.state['cur_merit_f']
        self.state['search_rhs'] = 0
        self.state['active_constraints'] = active_constraints

        self.timing_stats["step_computation"] += time.time() - step_comp_start

        # Apply step
        self.step_size = self.step_size_init
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        d_p_i_start = 0

        for p in group['params']:
            d_p_i_end = d_p_i_start + len(p.view(-1))
            d_p = d[d_p_i_start:d_p_i_end].reshape(p.shape)
            p.data.add_(d_p, alpha=self.step_size)
            d_p_i_start = d_p_i_end

        # Save step information
        self.state['d'] = d.clone()
        if active_constraints and len(active_constraints) > 0:
            self.state['J_save'] = J[active_constraints[0]:active_constraints[0]+1].clone()

        # Update timing stats
        self.timing_stats["total_linalg"] += time.time() - start_time_total
        self.timing_stats["calls"] += 1

        # Cleanup
        del ls_matrix_full, H
        torch.cuda.empty_cache()

        return None
