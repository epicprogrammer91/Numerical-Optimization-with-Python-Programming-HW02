import numpy as np
from src.unconstrained_min import LSM

# Constrained Optimization Class
class ConstrainedMin:

    def __init__(self, obj_tol=10 ** (-12), param_tol=10 ** (-8)):
        self.mu = 10  #log-barrier parameter
        self.t = 1    #barrier parameter
        self.epsilon = 10 ** (-12)
        self.beta_1 = 0.09  # wolfe 1 condition constant
        self.rho = 0.5 #line-search parameter

        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.f = None
        self.ineq_constraints = None
        self.eq_constraints_mat = None
        self.eq_constraints_rhs = None
        self.x0 = None
        self.A = None
        self.m = None

        self.gradients = {}  # dictionary of gradient at specific point in function
        self.hessians = {}  # dictionary of hessian at specific point in function
        self.f_at_x = {}  # dictionary of function value at specific point in function
        self.x_k_history = []
        self.results = {'x_k': None, 'f_at_x_k': None}

    def generate_results(self):
        f_at_x = []
        for x_k in self.x_k_history:
            f_at_x.append(self.f_at_x[str(x_k)])

        self.results['x_k'] = self.x_k_history.copy()
        self.results['f_at_x_k'] = f_at_x.copy()

    def print_to_console(self, i, x_i, f_x_i):
        print('outer',i, '\t', x_i, '\t', f_x_i)

    def check_termination(self):
        return self.m / self.t < self.epsilon

    def check_param_tol(self, x_k, x_k_plus_1):
        dist = np.linalg.norm(x_k - x_k_plus_1)
        terminate = (dist < self.param_tol)
        return terminate

    #barrier function, phi of x ...
    def phi_of_x(self, x):
        value = 0

        for func in self.ineq_constraints:
            f, g, h = func(x)
            value += np.log(-f)
        value = -value

        return value

    #gradient of phi of x ...
    def g_phi_of_x(self, x):
        value = 0

        for func in self.ineq_constraints:
            f, g, h = func(x)
            value += ((1 / (-f)) * g)

        return value

    #hessian of phi of x ...
    def h_phi_of_x(self, x):
        value = 0

        for func in self.ineq_constraints:
            f, g, h = func(x)
            value += ((1 / (f ** 2)) * (np.outer(g,g))) + ((1 / (-f)) * h)

        return value

    def outer_iteration(self, x_k):
        #step 1: start with self.t = 1
        if self.A is not None:
            #step 2: solve tf_o(x) + phi(x) s.t. Ax = b (constrained case):
            min_x = self.newtons_method(x_k)
        else:
            # step 2: solve tf_o(x) + phi(x) (unconstrained case) :
            lsm = LSM(self.unconstrained_objective_func, x0=x_k)
            min_x, f_at_x, _ = lsm.newton()

        #step 3: check termination:
        if(self.check_termination()):
            return min_x, True
        #step 4: t := mu * t
        self.t = self.mu * self.t
        return min_x, False


    def unconstrained_objective_func(self, x, eval_Hessian=False):
        f = None  # scalar function value
        g = None  # gradient vector
        h = None  # Hessian matrix

        f_o, g_o, h_o = self.f(x, eval_Hessian=eval_Hessian)

        phi_of_x_k = self.phi_of_x(x)
        g_phi_of_x_k = self.g_phi_of_x(x)
        h_phi_of_x_k = self.h_phi_of_x(x)

        f = (self.t * f_o) + phi_of_x_k
        g = (self.t * g_o) + g_phi_of_x_k
        if eval_Hessian:
            h = (self.t * h_o) + h_phi_of_x_k

        return f, g, h

    def interior_pt(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):

        self.x0 = x0
        self.f = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.A = self.eq_constraints_mat
        self.m = len(self.ineq_constraints)

        iteration = 0
        x_k = x0
        self.x_k_history.append(x_k)

        while(True):
            x_k_plus_1, terminate = self.outer_iteration(x_k)
            self.x_k_history.append(x_k_plus_1)
            if self.A is not None:
                self.print_to_console(iteration, x_k, self.f_at_x[str(x_k)])
            else:
                self.f_at_x[str(x_k)],_,_ = self.f(x_k)
                self.print_to_console(iteration, x_k, self.f_at_x[str(x_k)])
            if terminate:
                self.f_at_x[str(x_k_plus_1)], _, _ = self.f(x_k_plus_1)
                self.generate_results()
                return x_k_plus_1, self.f_at_x[str(x_k_plus_1)]

            iteration += 1
            x_k = x_k_plus_1


    #Newton's method for convex problems with equality constraints:
    def newtons_method(self, x_k):
        iteration = 0
        while(True):
            x_k_plus_1, terminate = self.newton_step(x_k)
            if terminate:
                break
            if self.check_param_tol(x_k, x_k_plus_1):
                break

            x_k = x_k_plus_1
            iteration += 1

        return x_k_plus_1


    # compute the Newton step p_nt by solving the local KKT system
    def newton_step(self, x_k):
        # given x_k that's feasible: Ax_k = b  ...

        f, g, h = self.unconstrained_objective_func(x_k, eval_Hessian=True)
        f_o, _,_= self.f(x_k, eval_Hessian=True)

        self.hessians[str(x_k)] = h
        self.gradients[str(x_k)] = g
        self.f_at_x[str(x_k)] = f_o

        left_matrix, right_vector = self.get_local_KKT_system(self.gradient_f(x_k), self.hessian_f(x_k), self.A)

        p_w = np.linalg.solve(left_matrix, right_vector)
        # direction obtained by extracting p from above vector and ignoring the dual for the local KKT:
        d_k = p_w[0:self.hessian_f(x_k).shape[1]]  # d_k can also be called p_nt

        # check termination criterion with Newton decrement...
        if (self.newton_decrement(self.hessian_f(x_k), d_k)):
            return x_k, True

        alpha_k = self.line_search(x_k, d_k)
        x_k_plus_1 = x_k + (alpha_k * d_k)  # follows direction d_k with step size alpha_k

        return x_k_plus_1, False


    def get_local_KKT_system(self, f_o_grad, f_o_hessian, A):
        if A.ndim == 1:
            A_T = A[np.newaxis].T
        else:
            A_T = A.T

        left_over_size = f_o_hessian.shape[1] + A_T.shape[1] - A.shape[0]
        matrix_zero_block = np.zeros((left_over_size))
        left_matrix = np.block([
            [f_o_hessian, A_T],
            [A, matrix_zero_block]
        ])

        right_vector = np.block([-f_o_grad, np.zeros(f_o_hessian.shape[0] + matrix_zero_block.shape[0] - f_o_grad.shape[0])])

        return left_matrix, right_vector


    def hessian_f(self, x_k):
        return self.hessians[str(x_k)]

    def gradient_f(self, x_k):
        return self.gradients[str(x_k)]

    def newton_decrement(self, H, p_nt):
        lambda_squared = p_nt.dot(H @ p_nt)  # quadratic form of the Hessian operating on the Newton direction
        terminate = ((1 / 2)*lambda_squared < self.obj_tol)
        return terminate

    def line_search(self, x_k, d_k, rho=None):
        if rho is None:
            rho = self.rho
        assert 0 < rho < 1
        alpha_k = 1  # Current candidate
        while not self.first_wolfe_condition_validity_check(x_k, alpha_k, d_k):
            alpha_k = rho * alpha_k  # update to new alpha if first wolfe condition failed

        return alpha_k

    # forbidding steps that are too long
    def first_wolfe_condition_validity_check(self, x_k, alpha_k, d_k, beta_1=None):
        if beta_1 is None:
            beta_1 = self.beta_1
        assert 0 < beta_1 < 1
        valid = ((self.f(x_k + (alpha_k * d_k))[0]) <= (
                self.f(x_k)[0] + (alpha_k * beta_1 * self.gradient_f(x_k).dot(d_k))))
        return valid
