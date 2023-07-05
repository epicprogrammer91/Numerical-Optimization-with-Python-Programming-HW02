import numpy as np

# Line Search Minimization
class LSM:

    def __init__(self, f, x0=np.array([1., 1.]), obj_tol=10 ** (-12), param_tol=10 ** (-8), max_iter=100):
        self.f = f
        self.x0 = x0.T
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter  # if we reached max iteration value that means we failed (return False)

        self.gradients = {}  # dictionary of gradient at specific point in function
        self.hessians = {}  # dictionary of hessian at specific point in function
        self.f_at_x = {}  # dictionary of function value at specific point in function

        self.beta_1 = 0.01  # wolfe 1 condition constant
        self.rho = 0.5
        self.tau_increment = 0.01

        self.x_k_history = []
        self.B_k = None  # used for SR1 and BFGS: begin with either identity matrix or the Hessian at initial point
        self.results = {'x_k':None, 'f_at_x_k':None, 'method':None} #final iterations path results dictionary

    def print_to_console(self, i, x_i, f_x_i):
        print(i, '\t', x_i, '\t', f_x_i)

    def check_obj_tol(self, f_x_k, f_x_k_plus_1):
        terminate = (np.absolute(f_x_k_plus_1 - f_x_k) < self.obj_tol)
        return terminate

    def check_param_tol(self, x_k, x_k_plus_1):
        dist = np.linalg.norm(x_k - x_k_plus_1)
        terminate = (dist < self.param_tol)
        return terminate

    # preconditioned steepest descent algorithm iteration (in original variables)
    # matrix H_k is symmetric positive definite (only requirement), which defines the change of variables
    # H_k makes change of variable implicit
    # H_k^-1 is called the preconditioner
    # Each specific value of H_k will induce a different algorithm
    def descent_step(self, x_k, H_k, newton=False):

        d_k = np.linalg.solve(H_k, -self.gradient_f(x_k))  # direction
        d_k_all_zeros = not np.any(d_k)
        '''
        # below assertion is always the case when H_k is symmetric positive definite: ..
        if not d_k_all_zeros:
            assert self.gradient_f(x_k).dot(d_k) < 0  # assert d_k is a descent direction for function f at point x_k
        '''
        alpha_k = self.line_search(x_k, d_k)
        x_k_plus_1 = x_k + alpha_k * d_k  # follows direction d_k with step size alpha_k
        if newton:
            return x_k_plus_1, self.newton_decrement(H_k, d_k)
        return x_k_plus_1, d_k_all_zeros

    def BFGS_step(self, x_k, x_k_plus_1):

        s_k = x_k_plus_1 - x_k
        y_k = self.gradients[str(x_k_plus_1)] - self.gradients[str(x_k)]  # difference btwn the previous two gradients
        B_k = self.B_k
        if not np.any(y_k) or not np.any(s_k) or not np.any(B_k):
            return None, None

        B_k_plus_1 = B_k - ((B_k @ s_k @ s_k.dot(B_k)) / (s_k.dot(B_k) @ s_k)) + ((y_k @ y_k.T) / (y_k.dot(s_k)))

        '''
        tau = 0
        I = np.identity(self.gradients[str(x_k)].size)
        while True:
            try:
                B_k_plus_1 = B_k_plus_1 + (tau * I)  # strengthen H_k's diagonal
                np.linalg.cholesky(B_k_plus_1)
                # x_k_plus_1 = self.descent_step(x_k, H_k, newton=True)
                break
            except np.linalg.LinAlgError:  # thrown when H_k is not symmetric positive definite: ...
                tau += self.tau_increment
                continue
        '''
        self.B_k = B_k_plus_1

        H_k = self.B_k
        x_k = x_k_plus_1
        x_k_plus_1, d_k_all_zeros = self.descent_step(x_k, H_k)

        return x_k_plus_1, d_k_all_zeros

    def BFGS(self):
        self.results['method'] = 'BFGS'
        success = True

        x_k1 = self.x0
        f, g, h = self.f(x_k1)
        self.hessians[str(x_k1)] = h
        self.gradients[str(x_k1)] = g
        self.f_at_x[str(x_k1)] = f
        self.x_k_history.append(x_k1)
        self.print_to_console(0, x_k1, self.f_at_x[str(x_k1)])

        I = np.identity(self.gradients[str(x_k1)].size)
        x_k2, _ = self.descent_step(x_k1, I)
        f, g, h = self.f(x_k2, eval_Hessian=True)
        self.hessians[str(x_k2)] = h
        self.gradients[str(x_k2)] = g
        self.f_at_x[str(x_k2)] = f
        self.x_k_history.append(x_k2)
        self.print_to_console(1, x_k2, self.f_at_x[str(x_k2)])

        self.B_k = h

        for i in range(2, self.max_iter + 2):
            x_k2_plus_1, d_all_zeros = self.BFGS_step(x_k1, x_k2)
            if x_k2_plus_1 is None:
                success = False
                self.generate_results()
                return x_k2, self.f_at_x[str(x_k2)], success
                
            f, g, h = self.f(x_k2_plus_1)
            self.hessians[str(x_k2_plus_1)] = h
            self.gradients[str(x_k2_plus_1)] = g
            self.f_at_x[str(x_k2_plus_1)] = f

            self.x_k_history.append(x_k2_plus_1)
            self.print_to_console(i, x_k2_plus_1, self.f_at_x[str(x_k2_plus_1)])
            if self.check_param_tol(x_k2, x_k2_plus_1):
                break
            if self.check_obj_tol(self.f_at_x[str(x_k2)], self.f_at_x[str(x_k2_plus_1)]):
                break
            if d_all_zeros:
                break
            if i == self.max_iter:
                success = False
            x_k1 = x_k2
            x_k2 = x_k2_plus_1
        self.generate_results()

        return x_k2_plus_1, self.f_at_x[str(x_k2_plus_1)], success

    def SR1_step(self, x_k, x_k_plus_1):

        s_k = x_k_plus_1 - x_k
        y_k = self.gradients[str(x_k_plus_1)] - self.gradients[str(x_k)]  # difference btwn the previous two gradients
        B_k = self.B_k

        if not np.any(s_k) or not np.any(y_k):
            return None, None

        # Update the Hessian approximation
        y_k_minus_B_k_s_k = y_k - np.dot(B_k, s_k)
        denom = np.dot(y_k_minus_B_k_s_k, s_k)

        if denom > 1e-8:
            B_k += np.outer(y_k_minus_B_k_s_k, y_k_minus_B_k_s_k) / denom

        '''
        tau = 0
        I = np.identity(self.gradients[str(x_k)].size)
        while True:
            try:
                B_k = B_k + (tau * I)  # strengthen H_k's diagonal
                np.linalg.cholesky(B_k)
                #x_k_plus_1 = self.descent_step(x_k, H_k, newton=True)
                break
            except np.linalg.LinAlgError:  # thrown when H_k is not symmetric positive definite: ...
                tau += self.tau_increment
                continue

        '''
        self.B_k = B_k
        H_k = self.B_k
        x_k_plus_1, d_k_all_zeros = self.descent_step(x_k, H_k)
        return x_k_plus_1, d_k_all_zeros

    def SR1(self):
        self.results['method'] = 'SR1'
        success = True

        x_k1 = self.x0
        f, g, h = self.f(x_k1)
        self.hessians[str(x_k1)] = h
        self.gradients[str(x_k1)] = g
        self.f_at_x[str(x_k1)] = f
        self.x_k_history.append(x_k1)
        self.print_to_console(0, x_k1, self.f_at_x[str(x_k1)])

        I = np.identity(self.gradients[str(x_k1)].size)
        x_k2, _ = self.descent_step(x_k1, I)
        f, g, h = self.f(x_k2 , eval_Hessian=True)
        self.hessians[str(x_k2)] = h
        self.gradients[str(x_k2)] = g
        self.f_at_x[str(x_k2)] = f
        self.x_k_history.append(x_k2)
        self.print_to_console(1, x_k2, self.f_at_x[str(x_k2)])
        self.B_k = h

        for i in range(2, self.max_iter + 2):
            x_k2_plus_1, d_all_zeros = self.SR1_step(x_k1, x_k2)
            if x_k2_plus_1 is None:
                self.generate_results()
                success = False
                return x_k2, self.f_at_x[str(x_k2)], success

            f, g, h = self.f(x_k2_plus_1)
            self.hessians[str(x_k2_plus_1)] = h
            self.gradients[str(x_k2_plus_1)] = g
            self.f_at_x[str(x_k2_plus_1)] = f

            self.x_k_history.append(x_k2_plus_1)
            self.print_to_console(i, x_k2_plus_1, self.f_at_x[str(x_k2_plus_1)])
            if self.check_param_tol(x_k2, x_k2_plus_1):
                break
            if self.check_obj_tol(self.f_at_x[str(x_k2)], self.f_at_x[str(x_k2_plus_1)]):
                break
            if d_all_zeros:
                break
            if i == self.max_iter:
                success = False
            x_k1 = x_k2
            x_k2 = x_k2_plus_1
        self.generate_results()
        return x_k2_plus_1, self.f_at_x[str(x_k2_plus_1)], success

    def gradient_descent_step(self, x_k):
        I = np.identity(self.gradients[str(x_k)].size)
        H_k = I
        x_k_plus_1, d_k_all_zeros = self.descent_step(x_k, H_k)
        return x_k_plus_1, d_k_all_zeros

    def generate_results(self):
        f_at_x = []
        for x_k in self.x_k_history:
            f_at_x.append(self.f_at_x[str(x_k)])

        self.results['x_k'] = self.x_k_history.copy()
        self.results['f_at_x_k'] = f_at_x.copy()
        #print('self.results',self.results)

    def gradient_descent(self):
        self.results['method'] = 'GD'
        success = True
        x_k = self.x0
        f, g, h = self.f(x_k)
        self.hessians[str(x_k)] = h
        self.gradients[str(x_k)] = g
        self.f_at_x[str(x_k)] = f

        self.x_k_history.append(x_k)
        self.print_to_console(0, x_k, self.f_at_x[str(x_k)])
        for i in range(1, self.max_iter + 1):
            x_k_plus_1, d_k_all_zeros = self.gradient_descent_step(x_k)
            f, g, h = self.f(x_k_plus_1)
            self.hessians[str(x_k_plus_1)] = h
            self.gradients[str(x_k_plus_1)] = g
            self.f_at_x[str(x_k_plus_1)] = f

            self.x_k_history.append(x_k_plus_1)
            self.print_to_console(i, x_k_plus_1, self.f_at_x[str(x_k_plus_1)])
            if self.check_param_tol(x_k, x_k_plus_1):
                break
            if self.check_obj_tol(self.f_at_x[str(x_k)], self.f_at_x[str(x_k_plus_1)]):
                break
            if d_k_all_zeros:
                break
            if i == self.max_iter:
                success = False
            x_k = x_k_plus_1

        self.generate_results()
        return x_k_plus_1, self.f_at_x[str(x_k_plus_1)], success

    # The descent method framework is providing safe-guards to the Newton's method
    # and brings it in a neighbourhood where Newton's method can actually use its full power.
    # It is a fast and reliable algorithm.
    def newton_step(self, x_k):

        H_k = self.hessians[str(x_k)]
        # loop ensures that H_k is symmetric positive definite: ...
        tau = 0
        I = np.identity(self.gradients[str(x_k)].size)
        while True:
            try:
                H_k = H_k + (tau * I)  # strengthen H_k's diagonal
                np.linalg.cholesky(H_k)
                x_k_plus_1 = self.descent_step(x_k, H_k, newton=True)
                break
            except np.linalg.LinAlgError:  # thrown when H_k is not symmetric positive definite: ...
                tau += self.tau_increment
                continue
                pass

        return x_k_plus_1

    def newton(self):
        self.results['method'] = 'Newton'
        success = True
        x_k = self.x0
        f, g, h = self.f(x_k, eval_Hessian=True)
        self.hessians[str(x_k)] = h
        self.gradients[str(x_k)] = g
        self.f_at_x[str(x_k)] = f

        self.x_k_history.append(x_k)
        self.print_to_console(0, x_k, self.f_at_x[str(x_k)])
        for i in range(1, self.max_iter + 1):
            x_k_plus_1, terminate = self.newton_step(x_k)
            f, g, h = self.f(x_k_plus_1, eval_Hessian=True)
            self.hessians[str(x_k_plus_1)] = h
            self.gradients[str(x_k_plus_1)] = g
            self.f_at_x[str(x_k_plus_1)] = f

            self.x_k_history.append(x_k_plus_1)
            self.print_to_console(i, x_k_plus_1, self.f_at_x[str(x_k_plus_1)])
            if self.check_param_tol(x_k, x_k_plus_1):
                break
            if terminate:
                break
            if i == self.max_iter:
                success = False
            x_k = x_k_plus_1
        self.generate_results()
        return x_k_plus_1, self.f_at_x[str(x_k_plus_1)], success


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

    def hessian_f(self, x_k):
        return self.hessians[str(x_k)]

    def gradient_f(self, x_k):
        return self.gradients[str(x_k)]

    # forbidding steps that are too long
    def first_wolfe_condition_validity_check(self, x_k, alpha_k, d_k, beta_1=None):
        if beta_1 is None:
            beta_1 = self.beta_1
        assert 0 < beta_1 < 1
        valid = ( (self.f(x_k + (alpha_k * d_k))[0]) <= (self.f(x_k)[0] + (alpha_k * beta_1 * self.gradient_f(x_k).dot(d_k))) )
        return valid

    '''
    # forbidding steps that are too short    
    def second_wolfe_condition_validity_check(self, x_k, alpha_k, d_k, beta_2=self.beta_2):
        assert 0 < beta_2 < 1
        valid = (self.gradient_f(x_k + (alpha_k*d_k)).dot(d_k)) >= (beta_2 * self.gradient_f(x_k).dot(d_k))
        return valid
    '''
