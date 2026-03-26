import torch
from torch.optim.optimizer import Optimizer, required
import time
from config import get_args

args = get_args()


class StochasticSQP(Optimizer):

    sigma = 0.5
    eta = 1e-4  # line search parameter
    buffer = 0
    eps = 1e-6

    def __init__(self, params, lr=required,
                 epsilon=1e-5,
                 upper_bound=[1e-5, 1e-5, 1e-5, 1e-5],  
                 lower_bound=[1e-5, 1e-5, 1e-5, 1e-5],
                 n_parameters=0,
                 n_constrs=0,
                 merit_param_init=1,
                 ratio_param_init=1,
                 step_size_decay=0.5,
                 step_opt=1,
                 problem=None,
                 mu=100,
                 beta2=0.999):
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
        
        # Initialize timing statistics
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

    def initialize_param(self, initial_value=1):
        # Update parameters
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        for p in group['params']:
            p.data.add_(initial_value, alpha=1)
        return
    
    def print_timing_stats(self):
        """Print the timing statistics collected during optimization."""
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
        
        # This was causing the error - removed reference to args.train_size
        print("=======================================\n")

    def step(self, closure=None):
        """Performs a single optimization step."""
        start_time_total = time.time()
        
        # Extract values from state
        J = self.state['J']  
        g = self.state['g']
        c = self.state['c']
        f = self.state['f']
        
        # Initialize active constraints list
        active_constraints = []
        
        # Update state (e.g. for adaptive scaling)
        if 'iter' not in self.state:
            self.state['iter'] = 0
            self.state['g_square_sum'] = ((1 - self.beta2) * g**2)
        else:
            self.state['iter'] += 1
            self.state['g_square_sum'] = (self.beta2 * self.state['g_square_sum'] +
                                          (1 - self.beta2) * g**2)

        # Start timing matrix construction
        matrix_construction_start = time.time()
        
        H_diag = torch.sqrt(self.state['g_square_sum'] + self.mu)
        H = torch.diag(H_diag)
        zeros = torch.zeros(self.n_constrs, self.n_constrs, device=J.device)
        
        # Full system matrix (used for multiple constraints)
        ls_matrix_full = torch.cat(( 
            torch.cat((H, torch.transpose(J, 0, 1)), dim=1),
            torch.cat((J, zeros), dim=1)
        ), dim=0)
        
        matrix_construction_time = time.time() - matrix_construction_start
        self.timing_stats["matrix_construction"] += matrix_construction_time

        # Common threshold to consider a bound as "extreme"
        THRESH = 1e12

        # Helper function to solve the full linear system for the given adjusted constraints
        def solve_system(c_adj):
            """Solves the full linear system for the given adjusted constraints c_adj."""
            solve_start = time.time()
            ls_rhs_adj = -torch.cat((g, c_adj), dim=0)
            system_solution_adj = torch.linalg.solve(ls_matrix_full, ls_rhs_adj)
            d_adj = system_solution_adj[:self.n_parameters]
            y_adj = system_solution_adj[self.n_parameters:]
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            return d_adj, y_adj, solve_time

        # Helper function to solve a system with a subset of constraints
        def solve_system_subset(c_adj_val, constraint_indices):
            """
            Solves a system with a subset of constraints.
            constraint_indices: list of indices to include
            """
            solve_start = time.time()
            n_subset = len(constraint_indices)
            J_sub = J[constraint_indices]  # shape (n_subset, n_parameters)
            zeros_sub = torch.zeros(n_subset, n_subset, device=J.device)
            ls_matrix_sub = torch.cat((
                torch.cat((H, J_sub.T), dim=1),
                torch.cat((J_sub, zeros_sub), dim=1)
            ), dim=0)
            ls_rhs_sub = -torch.cat((g, c_adj_val), dim=0)
            system_solution_sub = torch.linalg.solve(ls_matrix_sub, ls_rhs_sub)
            d_sub = system_solution_sub[:self.n_parameters]
            y_sub = system_solution_sub[self.n_parameters:self.n_parameters+n_subset]
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            return d_sub, y_sub, solve_time

        # Helper function to check if a solution is feasible
        def is_feasible(d_val, constraints_to_check=None):
            """Check if a direction d_val satisfies all constraints or specified ones"""
            check_start = time.time()
            if constraints_to_check is None:
                constraints_to_check = range(self.n_constrs)
                
            for i in constraints_to_check:
                constr_val = c[i] + torch.matmul(J[i], d_val)
                if hasattr(self, 'lower_bound') and i < len(self.lower_bound) and self.lower_bound[i] < THRESH:
                    if constr_val < -self.lower_bound[i]:
                        check_time = time.time() - check_start
                        self.timing_stats["constraint_checking"] += check_time
                        return False
                if hasattr(self, 'upper_bound') and i < len(self.upper_bound) and self.upper_bound[i] < THRESH:
                    if constr_val > self.upper_bound[i]:
                        check_time = time.time() - check_start
                        self.timing_stats["constraint_checking"] += check_time
                        return False
            
            check_time = time.time() - check_start
            self.timing_stats["constraint_checking"] += check_time
            return True

        # Helper function to create a full y vector from a subset solution
        def create_full_y(y_sub, active_indices):
            """Create full y vector with zeros for inactive constraints"""
            y_full = torch.zeros(self.n_constrs, device=y_sub.device)
            for idx, i in enumerate(active_indices):
                y_full[i] = y_sub[idx]
            return y_full

        def objective(d, y):
            """Computes the objective value for a given d and y."""
            eval_start = time.time()
            obj_val = torch.matmul(g, d) + 0.5 * torch.matmul(d, torch.matmul(H, d))
            eval_time = time.time() - eval_start
            self.timing_stats["candidate_evaluation"] += eval_time
            return obj_val

        # Default saved Jacobian is the full one.
        J_save = J.clone()
        
        # Case: No constraints.
        if self.n_constrs == 0:
            solve_start = time.time()
            d = torch.linalg.solve(H, -g)
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            y = None
            active_constraints = []

        # Case: One constraint.
        elif self.n_constrs == 1:
            # First try unconstrained solution
            solve_start = time.time()
            d_m = torch.linalg.solve(H, -g)
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            
            feasible_m = abs(c + torch.matmul(J, d_m)) <= self.epsilon

            if feasible_m:
                d, y = d_m, None
                active_constraints = []
            else:
                c_adj_upper = (c - self.epsilon).to(args.device)
                c_adj_lower = (c + self.epsilon).to(args.device)
                d_u, y_u, _ = solve_system(c_adj_upper)
                d_l, y_l, _ = solve_system(c_adj_lower)
                obj_u = objective(d_u, y_u)
                obj_l = objective(d_l, y_l)
                obj_values = torch.tensor([obj_u, obj_l], device=args.device)
                min_obj_idx = torch.argmin(obj_values)
                if min_obj_idx == 0:
                    d, y = d_u, y_u
                    active_constraints = [0]  # Upper bound is active
                elif min_obj_idx == 1:
                    d, y = d_l, y_l
                    active_constraints = [0]  # Lower bound is active

        # Case: Two constraints with upper and lower bounds
        elif self.n_constrs == 2:
            # Step 1: Try the unconstrained solution first
            solve_start = time.time()
            d_mm = torch.linalg.solve(H, -g)
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            
            if is_feasible(d_mm):
                d, y = d_mm, None
                active_constraints = []
            else:
                # Step 2: Try single active constraint solutions
                c_adj_upper = torch.zeros_like(c).to(args.device)
                c_adj_lower = torch.zeros_like(c).to(args.device)
                c_adj_upper[0] = c[0] - self.upper_bound[0]
                c_adj_lower[0] = c[0] + self.lower_bound[0]
                c_adj_upper[1] = c[1] - self.upper_bound[1]
                c_adj_lower[1] = c[1] + self.lower_bound[1]
                
                single_constraint_candidates = []
                
                # Try constraint 1 upper bound if not extreme
                if self.upper_bound[1] < THRESH:
                    try:
                        d_mu, y_mu, _ = solve_system_subset(c_adj_val=c_adj_upper[1:2],
                                                        constraint_indices=[1])
                        if is_feasible(d_mu, [0]):  # Check if feasible for constraint 0
                            y_mu_full = create_full_y(y_mu, [1])
                            single_constraint_candidates.append((d_mu, y_mu_full, objective(d_mu, y_mu), [1], 'upper'))
                    except Exception:
                        pass
                
                # Try constraint 1 lower bound if not extreme
                if self.lower_bound[1] < THRESH:
                    try:
                        d_ml, y_ml, _ = solve_system_subset(c_adj_val=c_adj_lower[1:2],
                                                        constraint_indices=[1])
                        if is_feasible(d_ml, [0]):  # Check if feasible for constraint 0
                            y_ml_full = create_full_y(y_ml, [1])
                            single_constraint_candidates.append((d_ml, y_ml_full, objective(d_ml, y_ml), [1], 'lower'))
                    except Exception:
                        pass

                # Try constraint 0 upper bound if not extreme
                if self.upper_bound[0] < THRESH:
                    try:
                        d_um, y_um, _ = solve_system_subset(c_adj_val=c_adj_upper[0:1],
                                                        constraint_indices=[0])
                        if is_feasible(d_um, [1]):  # Check if feasible for constraint 1
                            y_um_full = create_full_y(y_um, [0])
                            single_constraint_candidates.append((d_um, y_um_full, objective(d_um, y_um), [0], 'upper'))
                    except Exception:
                        pass
                
                # Try constraint 0 lower bound if not extreme
                if self.lower_bound[0] < THRESH:
                    try:
                        d_lm, y_lm, _ = solve_system_subset(c_adj_val=c_adj_lower[0:1],
                                                        constraint_indices=[0])
                        if is_feasible(d_lm, [1]):  # Check if feasible for constraint 1
                            y_lm_full = create_full_y(y_lm, [0])
                            single_constraint_candidates.append((d_lm, y_lm_full, objective(d_lm, y_lm), [0], 'lower'))
                    except Exception:
                        pass

                if single_constraint_candidates:
                    # If we have feasible solutions with one active constraint, take the best one
                    best_candidate = min(single_constraint_candidates, key=lambda x: x[2])
                    d, y = best_candidate[0], best_candidate[1]
                    active_constraints = best_candidate[3]
                    # Save the Jacobian row for the active constraint
                    J_save = J[active_constraints[0]:active_constraints[0]+1].clone()
                else:
                    # Step 3: Try solutions with two active constraints
                    double_constraint_candidates = []
                    
                    # Both upper bounds
                    if self.upper_bound[0] < THRESH and self.upper_bound[1] < THRESH:
                        try:
                            d_uu, y_uu, _ = solve_system(c_adj_upper)
                            double_constraint_candidates.append((d_uu, y_uu, objective(d_uu, y_uu), [0, 1], 'upper-upper'))
                        except Exception:
                            pass
                    
                    # Both lower bounds
                    if self.lower_bound[0] < THRESH and self.lower_bound[1] < THRESH:
                        try:
                            d_ll, y_ll, _ = solve_system(c_adj_lower)
                            double_constraint_candidates.append((d_ll, y_ll, objective(d_ll, y_ll), [0, 1], 'lower-lower'))
                        except Exception:
                            pass
                    
                    # Upper for 0, lower for 1
                    if self.upper_bound[0] < THRESH and self.lower_bound[1] < THRESH:
                        try:
                            c_adj_ul = torch.cat((c_adj_upper[0:1], c_adj_lower[1:2]), dim=0)
                            d_ul, y_ul, _ = solve_system(c_adj_ul)
                            double_constraint_candidates.append((d_ul, y_ul, objective(d_ul, y_ul), [0, 1], 'upper-lower'))
                        except Exception:
                            pass
                    
                    # Lower for 0, upper for 1
                    if self.lower_bound[0] < THRESH and self.upper_bound[1] < THRESH:
                        try:
                            c_adj_lu = torch.cat((c_adj_lower[0:1], c_adj_upper[1:2]), dim=0)
                            d_lu, y_lu, _ = solve_system(c_adj_lu)
                            double_constraint_candidates.append((d_lu, y_lu, objective(d_lu, y_lu), [0, 1], 'lower-upper'))
                        except Exception:
                            pass

                    if double_constraint_candidates:
                        # Take the best solution with two active constraints
                        best_candidate = min(double_constraint_candidates, key=lambda x: x[2])
                        d, y = best_candidate[0], best_candidate[1]
                        active_constraints = best_candidate[3]
                    else:
                        # Fallback: use unconstrained with reduced step size
                        d, y = d_mm, None
                        self.step_size *= self.step_size_decay
                        active_constraints = []

        # Case: Four constraints with only upper bounds (no lower bounds)
        elif self.n_constrs == 4:
            # Step 1: Try the unconstrained solution first
            solve_start = time.time()
            d_mmmm = torch.linalg.solve(H, -g)
            solve_time = time.time() - solve_start
            self.timing_stats["system_solving"] += solve_time
            
            if is_feasible(d_mmmm):
                d, y = d_mmmm, None
                active_constraints = []
            else:
                # Build adjusted constraint vectors for upper bounds only
                c_adj_upper = torch.zeros_like(c).to(args.device)
                
                for i in range(self.n_constrs):
                    c_adj_upper[i] = c[i] - self.upper_bound[i]
                
                # Step 2: Try single active constraint solutions
                single_u_candidates = []
                
                for i in range(self.n_constrs):
                    # Only consider this constraint if its upper bound is not extreme
                    if self.upper_bound[i] < THRESH:
                        try:
                            d_single, y_single, _ = solve_system_subset(
                                c_adj_val=c_adj_upper[i:i+1],
                                constraint_indices=[i]
                            )
                            
                            # Check if this solution is feasible for all other constraints
                            other_constraints = [j for j in range(self.n_constrs) if j != i]
                            if is_feasible(d_single, other_constraints):
                                # Create full y vector with helper function
                                y_full = create_full_y(y_single, [i])
                                obj_val = objective(d_single, y_full)
                                single_u_candidates.append((d_single, y_full, obj_val, [i]))
                        except Exception:
                            pass
                
                if single_u_candidates:
                    # If we have feasible solutions with one active constraint, take the best one
                    best_candidate = min(single_u_candidates, key=lambda x: x[2])
                    d, y = best_candidate[0], best_candidate[1]
                    active_constraints = best_candidate[3]
                    # Save the Jacobian row for the active constraint
                    J_save = J[active_constraints[0]:active_constraints[0]+1].clone()
                else:
                    # Step 3: Try solutions with two active constraints
                    double_u_candidates = []
                    
                    for i in range(self.n_constrs):
                        for j in range(i+1, self.n_constrs):
                            # Only consider these constraints if their upper bounds are not extreme
                            if self.upper_bound[i] < THRESH and self.upper_bound[j] < THRESH:
                                try:
                                    d_double, y_double, _ = solve_system_subset(
                                        c_adj_val=torch.cat((c_adj_upper[i:i+1], c_adj_upper[j:j+1]), dim=0),
                                        constraint_indices=[i, j]
                                    )
                                    
                                    # Check if this solution is feasible for all other constraints
                                    other_constraints = [k for k in range(self.n_constrs) if k != i and k != j]
                                    if is_feasible(d_double, other_constraints):
                                        # Create full y vector with helper function
                                        y_full = create_full_y(y_double, [i, j])
                                        obj_val = objective(d_double, y_full)
                                        double_u_candidates.append((d_double, y_full, obj_val, [i, j]))
                                except Exception:
                                    pass
                    
                    if double_u_candidates:
                        # If we have feasible solutions with two active constraints, take the best one
                        best_candidate = min(double_u_candidates, key=lambda x: x[2])
                        d, y = best_candidate[0], best_candidate[1]
                        active_constraints = best_candidate[3]
                    else:
                        # Step 4: Try solutions with three active constraints
                        triple_u_candidates = []
                        
                        for i in range(self.n_constrs):
                            # Create a list of all indices except i
                            other_indices = [j for j in range(self.n_constrs) if j != i]
                            
                            # Check if all three constraints have non-extreme upper bounds
                            if all(self.upper_bound[j] < THRESH for j in other_indices):
                                try:
                                    d_triple, y_triple, _ = solve_system_subset(
                                        c_adj_val=torch.cat([c_adj_upper[j:j+1] for j in other_indices], dim=0),
                                        constraint_indices=other_indices
                                    )
                                    
                                    # Check if this solution is feasible for the remaining constraint
                                    if is_feasible(d_triple, [i]):
                                        # Create full y vector with helper function
                                        y_full = create_full_y(y_triple, other_indices)
                                        obj_val = objective(d_triple, y_full)
                                        triple_u_candidates.append((d_triple, y_full, obj_val, other_indices))
                                except Exception:
                                    pass
                        
                        if triple_u_candidates:
                            # If we have feasible solutions with three active constraints, take the best one
                            best_candidate = min(triple_u_candidates, key=lambda x: x[2])
                            d, y = best_candidate[0], best_candidate[1]
                            active_constraints = best_candidate[3]
                        else:
                            # Step 5: Try the solution with all four constraints active
                            if all(self.upper_bound[i] < THRESH for i in range(self.n_constrs)):
                                try:
                                    d_all, y_all, _ = solve_system(c_adj_upper)
                                    # All constraints are active
                                    d, y = d_all, y_all
                                    active_constraints = list(range(self.n_constrs))
                                except Exception:
                                    # Fallback: use unconstrained with reduced step size
                                    d, y = d_mmmm, None
                                    self.step_size *= self.step_size_decay
                                    active_constraints = []
                            else:
                                # Fallback: use unconstrained with reduced step size
                                d, y = d_mmmm, None
                                self.step_size *= self.step_size_decay
                                active_constraints = []
        
        # Start timing step computation
        step_comp_start = time.time()
        
        # Compute the norm of the step.
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
                self.trial_merit = ((1 - self.sigma) * c_norm_1) / gd_plus_max_dHd_0

            if self.merit_param > self.trial_merit:
                self.merit_param = self.trial_merit * (1 - self.eps)

            delta_q = -self.merit_param * (gd + 2 * torch.max(dHd, torch.tensor(0, device=args.device))) + c_norm_1
            self.trial_ratio = delta_q / (self.merit_param * self.norm_d ** 2)

            if self.ratio_param > self.trial_ratio:
                self.ratio_param = self.trial_ratio * (1 - self.eps)

        # Line Search step-size update.
        self.state['merit_param'] = self.merit_param
        self.state['cur_merit_f'] = self.merit_param * f + torch.linalg.norm(c, 1)
        self.state['phi_new'] = self.state['cur_merit_f']
        self.state['search_rhs'] = 0
        
        step_comp_time = time.time() - step_comp_start
        self.timing_stats["step_computation"] += step_comp_time
        
        # Store active constraints in state
        self.state['active_constraints'] = active_constraints
        
        alpha_pre = 0.0
        phi = self.merit_param * f + torch.linalg.norm(c, 1)
        self.ls_k = 0
        self.step_size = self.step_size_init

        for self.ls_k in range(1):
            assert len(self.param_groups) == 1
            group = self.param_groups[0]
            d_p_i_start = 0
            for p in group['params']:
                d_p_i_end = d_p_i_start + len(p.view(-1))
                d_p = d[d_p_i_start:d_p_i_end].reshape(p.shape)
                p.data.add_(d_p, alpha=self.step_size - alpha_pre)
                d_p_i_start = d_p_i_end

        # Save step information
        self.state['d'] = d.clone()
        if J_save is not None:
            self.state['J_save'] = J_save.clone()
            
        # Update total linear algebra time
        total_linalg_time = time.time() - start_time_total
        self.timing_stats["total_linalg"] += total_linalg_time
        self.timing_stats["calls"] += 1
        
        # Clean up to save memory
        del ls_matrix_full
        del H
        torch.cuda.empty_cache()

        return None