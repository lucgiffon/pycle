import numpy as np
import scipy.optimize
from pdfo import pdfo
from sklearn.linear_model import LinearRegression

from pycle.compressive_learning.generic.Solver import Solver
from pycle.utils.optim import ObjectiveValuesStorage
from pycle.utils.datasets import sample_ball


# 0.2 CL-OMP template (stores a *mixture model* and implements a generic OMP for it)
class CLOMP(Solver):
    """
    Template for Compressive Learning with Orthogonal Matching Pursuit (CL-OMP) solver,
    used to the CL problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2,
    where P_theta = sum_{k=1}^K is a weighted mixture composed of K components P_theta_k,
    hence the problem to solve becomes
        min_(alpha,theta_k) || sketch_weight * z - sum_k alpha_k*A_Phi(P_theta_k) ||_2.
    The CLOMP algorithm works by adding new elements to the mixture one by one.
    """

    def __init__(self, Phi, K, d_atom, bounds, opt_method="vanilla", dct_opt_method=None,
                 show_curves=False, sketch=None, sketch_weight=1., verbose=0):
        """
        :param Phi: a FeatureMap object
        :param K: int, target number of mixture components
        :param d_atom: dimension of an atom, should be determined by a child class
        :param sketch: the sketch to be fit (can be None)
        :param sketch_weight: float, a re-scaling factor for the data sketch (default 1)
        :param opt_method: str, one of ["vanilla", "pdfo", "simplex gradient"]
        """
        # refacc: preparer ces trois elements (theta et atoms)
        #  avec des np.empty plutot que de les reremplir a chaque fois avec des concatenations
        self.Theta = None
        self.Atoms = None
        self.alpha = None

        self.Jacobians = None
        self.current_sol = None
        self.current_sol_cost = None
        self.residual = None
        # Call parent class
        super(CLOMP, self).__init__(Phi, sketch, sketch_weight, verbose)

        # Set other values
        # refacc: le n_atoms
        self.K = K
        self.n_atoms = 0
        self.d_atom = d_atom

        # Initialize empty solution
        self.initialize_empty_solution()

        # Set bounds
        self.bounds = None
        self.set_bounds_atom(bounds)  # bounds for an atom

        # Other minor params
        self.minimum_atom_norm = 1e-15 * np.sqrt(self.d_atom)
        self.weight_lower_bound = 1e-9
        self.weight_upper_bound = 2
        self.step5_ftol = 1e-6

        # Attributes related to dfo opt
        self.opt_method = opt_method
        self.dct_opt_method = dct_opt_method or dict()

        self.radius_sample = 1.
        self.show_curves = show_curves
        # self.bounds_atom = []

    # Abtract methods
    # ===============
    # Methods that have to be instantiated by child classes

    # Sketch of a single atom
    def sketch_of_atom(self, theta_k, return_jacobian=False):
        """
        Computes and returns A_Phi(P_theta_k) for an atom P_theta_k.
        possibly with the jacobian, of size (d_atom,m)
        """
        assert theta_k.size == self.d_atom
        raise NotImplementedError
        # if return_jacobian:
        #     return sketch_of_atom, jacobian
        # else:
        #     return sketch_of_atom

    def set_bounds_atom(self, bounds):
        """
        Should set self.bounds_atom to a list of length d_atom of lower and upper bounds, i.e.,
            self.bounds_atom = [[lowerbound_1,upperbound_1], ..., [lowerbound_d_atom,upperbound_d_atom]]
        """
        self.bounds = bounds  # data bounds
        raise NotImplementedError
        # self.bounds_atom = None
        # return None

    def randomly_initialize_new_atom(self):
        raise NotImplementedError
        # return new_theta

    # Generic methods
    # ===============
    # They should always work, using the instances of the methods above
    def initialize_empty_solution(self):
        self.n_atoms = 0
        self.alpha = np.empty(0)  # (n_atoms,)-array, weigths of the mixture elements
        self.Theta = np.empty((0, self.d_atom))  # (n_atoms,d_atom)-array, all the found parameters in matrix form
        self.Atoms = np.empty(
            (self.Phi.m, 0))  # (m,n_atoms)-array, the sketch of the found parameters (m is sketch size)
        self.Jacobians = np.empty(
            (0, self.d_atom, self.Phi.m))  # (n_atoms,d_atom,m)-array, the jacobians of the residual wrt each atom
        self.current_sol = (self.alpha, self.Theta)  # Overwrite

    def compute_atoms_matrix(self, Theta=None, return_jacobian=False):
        """
        Computes the matrix of atoms from scratch (if no Theta given, uses current Theta)
        """
        if Theta is not None:
            _n_atoms, _Theta = Theta.shape[0], Theta
        else:
            _n_atoms, _Theta = self.n_atoms, self.Theta

        _A = np.empty((self.Phi.m, _n_atoms), dtype=complex)
        # todo rewrite this to not use a loop
        if return_jacobian:
            _jac = 1j * np.empty((_n_atoms, self.d_atom, self.Phi.m))
            for k, theta_k in enumerate(_Theta):
                _A[:, k], _jac[k, :, :] = self.sketch_of_atom(theta_k, return_jacobian=True)
            return _A, _jac
        else:
            for k, theta_k in enumerate(_Theta):
                _A[:, k] = self.sketch_of_atom(theta_k, return_jacobian=False)
            return _A

    def update_atoms(self, Theta=None, update_jacobian=False):
        """
        Update the Atoms matrix (a (n_atoms,m)-array containing the A_Phi(P_theta_k) vectors)
        - with current Theta (self.Theta) if no argument is provided
        - with the provided Theta argument if one is given (self.Theta will also be updated)
        """
        if Theta is not None:
            self.Theta = Theta  # If necessary, update Theta
        if update_jacobian:
            self.Atoms, self.Jacobians = self.compute_atoms_matrix(return_jacobian=True)
        else:
            self.Atoms = self.compute_atoms_matrix(return_jacobian=False)

    # Add/remove atoms
    def add_atom(self, new_theta):
        self.n_atoms += 1
        self.Theta = np.append(self.Theta, [new_theta], axis=0)  # np.r_[self.Theta,new_theta]
        self.Atoms = np.c_[self.Atoms, self.sketch_of_atom(new_theta)]

    def remove_atom(self, index_to_remove):
        self.n_atoms -= 1
        # refacc
        self.Theta = np.delete(self.Theta, index_to_remove, axis=0)
        self.Atoms = np.delete(self.Atoms, index_to_remove, axis=1)

    def replace_atom(self, index_to_replace, new_theta):
        self.Theta[index_to_replace] = new_theta
        self.Atoms[:, index_to_replace] = self.sketch_of_atom(new_theta)

    # Stack/de-stack the found atoms
    def _stack_sol(self, alpha=None, Theta=None):
        """Stacks *all* the atoms and their weights into one vector"""
        # refacc
        if (Theta is not None) and (alpha is not None):
            _Theta, _alpha = Theta, alpha
        else:
            _Theta, _alpha = self.Theta, self.alpha
        return np.r_[_Theta.reshape(-1), _alpha]

    def _destack_sol(self, p):
        assert p.shape[-1] == self.n_atoms * (self.d_atom + 1)
        if len(p.shape) == 1 or p.shape[0] == 1:
            p = p.squeeze()
            Theta = p[:self.d_atom * self.n_atoms].reshape(self.n_atoms, self.d_atom)
            alpha = p[-self.n_atoms:].reshape(self.n_atoms)
        else:
            # todo à corriger
            raise NotImplementedError
            Theta = p[:, :self.d_atom * self.n_atoms].reshape(-1, self.n_atoms, self.d_atom)
            alpha = p[:, -self.n_atoms:].reshape(-1, self.n_atoms)
        return alpha, Theta

    # Optimization subroutines
    def _get_coeffs_lin_reg(self, sample_points_X, fct_values_Y):
        clf = LinearRegression()
        # stack with a column of one to fit the setting of polynomial interpolation
        first_order_pol_X = np.hstack([np.ones((sample_points_X.shape[0], 1)), sample_points_X])
        clf.fit(first_order_pol_X, fct_values_Y)
        lin_param = clf.coef_[1:]

        return lin_param

    def get_norm_sketch_theta(self, sketch_theta):
        assert len(sketch_theta.shape) == 2, "sketch_theta should be in 2D (n_samples, dim)"

        norm_sketch_theta = np.linalg.norm(sketch_theta, axis=1)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        # if norm_sketch_theta < self.minimum_atom_norm:
        #     if self.verbose > 1:
        #         print(f'norm_sketch_theta is too small ({norm_sketch_theta}), changed to {self.minimum_atom_norm}.')
        #     norm_sketch_theta = self.minimum_atom_norm

        norm_sketch_theta = np.maximum(norm_sketch_theta, self.minimum_atom_norm)
        return norm_sketch_theta

    ## First subproblem: max atom correlation
    def maximize_atom_correlation(self, new_theta):
        if self.opt_method == "vanilla":
            fct_fun_grad = self._maximize_atom_correlation_fun_grad
            sol = scipy.optimize.minimize(fct_fun_grad,
                                          x0=new_theta,
                                          method='L-BFGS-B', jac=True,
                                          bounds=self.bounds_atom)
        elif self.opt_method == "pdfo":
            fct_fun_grad = self._get_residual_correlation_value
            nb_iter_max = self.dct_opt_method.get("nb_iter_max_step_1", 100)
            sol = pdfo(fct_fun_grad,
                       x0=new_theta,  # Start at current solution
                       bounds=self.bounds_atom,
                       options={'maxfev': nb_iter_max * new_theta.size}
                       )
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method}")

        if self.show_curves:
            ObjectiveValuesStorage().show()
            ObjectiveValuesStorage().clear()

        return sol.x

    def _maximize_atom_correlation_fun_grad(self, theta):
        """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
        # Firstly, compute A(P_theta)...
        sketch_theta, jacobian_theta = self.sketch_of_atom(theta, return_jacobian=True)

        # ... and its l2 norm
        norm_sketch_theta = self.get_norm_sketch_theta(sketch_theta.reshape(1, -1))

        # Evaluate the cost function
        fun = -np.real(np.vdot(sketch_theta, self.residual)) / norm_sketch_theta  # - to have a min problem

        # Secondly, get the Jacobian
        grad = (-np.real(jacobian_theta @ np.conj(self.residual)) / norm_sketch_theta
                + np.real(np.real(jacobian_theta @ np.conj(sketch_theta)) * np.vdot(sketch_theta, self.residual)) / (
                        norm_sketch_theta ** 3))

        return fun, grad

    def _maximize_atom_correlation_fun_grad_simplex_grad(self, theta):
        """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
        # todo factoriser le code avec la fonction du dessus

        # Firstly, compute A(P_theta)...
        if self.dct_opt_method["compute_oracle"]:
            sketch_theta, jacobian_theta = self.sketch_of_atom(theta, return_jacobian=True)
        else:
            sketch_theta = self.sketch_of_atom(theta, return_jacobian=False)

        # ... and its l2 norm
        norm_sketch_theta = self.get_norm_sketch_theta(sketch_theta.reshape(1, -1))

        cost_fun = lambda x: -np.real(np.dot(self.Phi(x).conj(), self.residual)) / self.get_norm_sketch_theta(
            self.Phi(x).reshape(-1, sketch_theta.shape[-1]))
        # Evaluate the cost function
        # fun_value = -np.real(np.vdot(sketch_theta, self.residual)) / norm_sketch_theta  # - to have a min problem
        fun_value = cost_fun(theta.reshape(1, -1))[0]

        grad = self.get_gradient_estimate_max_corr(theta, cost_fun)

        if self.dct_opt_method["compute_oracle"]:
            grad_oracle = (-np.real(jacobian_theta @ np.conj(self.residual)) / norm_sketch_theta
                           + np.real(
                        np.real(jacobian_theta @ np.conj(sketch_theta)) * np.vdot(sketch_theta, self.residual)) / (
                                   norm_sketch_theta ** 3))

        if self.show_curves:
            ObjectiveValuesStorage().add(fun_value, "fun_val max corr")

        return fun_value, grad

    def _get_residual_correlation_value(self, theta):
        """Computes the fun. value and grad. of step 1 objective: max_theta <A(P_theta),r> / <A(P_theta),A(P_theta)>"""
        # Firstly, compute A(P_theta)...
        sketch_theta = self.sketch_of_atom(theta, return_jacobian=False).reshape(1, -1)

        fun_value = (-np.real(np.dot(sketch_theta.conj(), self.residual)) /
                     self.get_norm_sketch_theta(sketch_theta.reshape(-1, sketch_theta.shape[-1])))[0]

        if self.show_curves:
            ObjectiveValuesStorage().add(fun_value, "fun_val max corr")

        return fun_value

    def get_gradient_estimate_max_corr(self, theta, obj_fun):
        """
        Find a linear approximation of the gradient in the max atom correlation problem of clomp
        """
        # todo remove this "100"
        sample_points_X = sample_ball(radius=self.radius_sample, npoints=self.dct_opt_method["nb_sample_point"] * 100, ndim=theta.size,
                                      center=theta)
        fct_values_Y = np.array([obj_fun(elm) for elm in sample_points_X]).squeeze()
        # todo could be accelerated by removing the for loop

        return self._get_coeffs_lin_reg(sample_points_X, fct_values_Y)

    ## Second subproblem: best non negative weights
    def find_optimal_weights(self, normalize_atoms=False):
        """
        2 Optimisation problem of clomp.

        Using the current atoms matrix, find the optimal weights with scipy's nnls
        """
        # Stack real and imaginary parts if necessary
        if np.any(np.iscomplex(self.Atoms)):  # True if complex sketch output
            _A = np.r_[self.Atoms.real, self.Atoms.imag]
            _z = np.r_[self.sketch_reweighted.real, self.sketch_reweighted.imag]
        else:
            _A = self.Atoms
            _z = self.sketch_reweighted

        # Normalize if necessary
        if normalize_atoms:
            norms = np.linalg.norm(self.Atoms, axis=0)
            norm_too_small = np.where(norms < self.minimum_atom_norm)[0]
            if norm_too_small.size > 0:  # Avoid division by zero
                if self.verbose > 1:
                    print(f'norm of some atoms is too small (min. {norms.min()}), changed to {self.minimum_atom_norm}.')
                norms[norm_too_small] = self.minimum_atom_norm
            _A = _A / norms

        # Use non-negative least squares to find optimal weights
        (_alpha, _) = scipy.optimize.nnls(_A, _z)
        return _alpha

    ## Third subproblem: finetune solution
    def minimize_cost_from_current_sol(self, ftol=None):
        if ftol is None:
            ftol = self.step5_ftol

        bounds_Theta_alpha = self.bounds_atom * self.n_atoms + [
            [self.weight_lower_bound, self.weight_upper_bound]] * self.n_atoms

        if self.opt_method == "vanilla":
            fct_fun_grad = self._minimize_cost_from_current_sol
            sol = scipy.optimize.minimize(fct_fun_grad,
                                          x0=self._stack_sol(),  # Start at current solution
                                          method='L-BFGS-B', jac=True,
                                          bounds=bounds_Theta_alpha, options={'ftol': ftol})
        elif self.opt_method == "pdfo":
            fct_fun_grad = self.get_global_cost
            init_x0 = self._stack_sol()
            nb_iter_max = self.dct_opt_method.get("nb_iter_max_step_5", 5)
            sol = pdfo(fct_fun_grad,
                       x0=init_x0,  # Start at current solution
                       bounds=bounds_Theta_alpha,
                       options={'maxfev': nb_iter_max * init_x0.size,
                                # 'rhoend': ftol
                                }
                       )
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method}")

        (self.alpha, self.Theta) = self._destack_sol(sol.x)

        if self.show_curves:
            ObjectiveValuesStorage().show()
            ObjectiveValuesStorage().clear()

    def _minimize_cost_from_current_sol(self, p):
        """
        Computes the fun. value and grad. of step 5 objective: min_alpha,Theta || z - alpha*A(P_Theta) ||_2,
        at the point given by p (stacked Theta and alpha), and updates the current sol to match.
        """
        # De-stack the parameter vector
        (_alpha, _Theta) = self._destack_sol(p.reshape(1, -1))

        # Update the weigths
        self.alpha = _alpha

        # Update the atom matrix and compute the Jacobians
        self.update_atoms(_Theta, update_jacobian=True)

        # Now that the solution is updated, update the residual
        self.residual = self.sketch_reweighted - self.sketch_of_solution()

        # Evaluate the cost function
        fun = np.linalg.norm(self.residual) ** 2

        # Evaluate the gradients
        grad = np.empty((self.d_atom + 1) * self.n_atoms)
        for k in range(self.n_atoms):  # Gradients of the atoms
            grad[k * self.d_atom:(k + 1) * self.d_atom] = -2 * self.alpha[k] * np.real(
                self.Jacobians[k] @ self.residual.conj())
        grad[-self.n_atoms:] = -2 * np.real(self.residual @ self.Atoms.conj())  # Gradient of the weights

        return fun, grad

    def _minimize_cost_from_current_sol_simplex_grad(self, p):
        """
        Computes the fun. value and grad. of step 5 objective: min_alpha,Theta || z - alpha*A(P_Theta) ||_2,
        at the point given by p (stacked Theta and alpha), and updates the current sol to match.
        """

        def eval_obj(x):
            # can only take on sample at a time
            (_alpha, _Theta) = self._destack_sol(x)
            sketch_of_solution = _alpha @ self.Phi(_Theta)
            # sketch_of_solution = self.Phi(_Theta).T @ _alpha
            r = self.sketch_reweighted - sketch_of_solution
            return np.linalg.norm(r, axis=-1) ** 2

        # De-stack the parameter vector
        (_alpha, _Theta) = self._destack_sol(p)

        # Update the weigths
        self.alpha = _alpha

        if self.dct_opt_method["compute_oracle"]:
            # Update the atom matrix and compute the Jacobians
            self.update_atoms(_Theta, update_jacobian=True)
        else:
            self.update_atoms(_Theta, update_jacobian=False)

        # Now that the solution is updated, update the residual
        self.residual = self.sketch_reweighted - self.sketch_of_solution()

        # Evaluate the cost function
        fun_val = eval_obj(p)

        if self.dct_opt_method["compute_oracle"]:
            fun_val_oracle = np.linalg.norm(self.residual) ** 2
            assert np.isclose(fun_val_oracle, fun_val)

        # Evaluate the gradients
        if self.dct_opt_method["compute_oracle"]:
            grad_oracle = np.empty((self.d_atom + 1) * self.n_atoms)
            for k in range(self.n_atoms):  # Gradients of the atoms
                grad_oracle[k * self.d_atom:(k + 1) * self.d_atom] = -2 * self.alpha[k] * np.real(
                    self.Jacobians[k] @ self.residual.conj())
            grad_oracle[-self.n_atoms:] = -2 * np.real(self.residual @ self.Atoms.conj())  # Gradient of the weights

        grad = self.get_gradient_estimate_finetuning(p, eval_obj)
        grad[-self.n_atoms:] = -2 * np.real(self.residual @ self.Atoms.conj())  # Gradient of the weights

        if self.show_curves:
            ObjectiveValuesStorage().add(fun_val, "fun_val finetuning")

        # return fun_val
        return fun_val, grad

    def get_global_cost(self, p):
        """
        Computes the fun. value of step 5 objective: min_alpha,Theta || z - alpha*A(P_Theta) ||_2,
        at the point given by p (stacked Theta and alpha).

        Can only take one solution p at a time.

        :param p Stacked Theta and alpha
        """
        (_alpha, _Theta) = self._destack_sol(p)
        sketch_of_solution = _alpha @ self.Phi(_Theta)
        # sketch_of_solution = self.Phi(_Theta).T @ _alpha
        r = self.sketch_reweighted - sketch_of_solution
        fun_val = np.linalg.norm(r, axis=-1) ** 2

        if self.show_curves:
            ObjectiveValuesStorage().add(fun_val, f"fun_val finetuning {p.shape}")

        return fun_val

    def get_gradient_estimate_finetuning(self, current_sol, obj_fun):
        """
        Find a linear approximation of the gradient in the finetuning problem of clomp
        """
        (_alpha, _Theta) = self._destack_sol(current_sol)
        nb_atoms = len(_alpha)

        # todo remove this "100"
        sample_points_X = sample_ball(radius=self.radius_sample, npoints=self.nb_sample_point * 100,
                                      ndim=current_sol.size, center=current_sol)
        sample_points_X[:, -nb_atoms:] = _alpha
        # todo remove alpha from the input if performance is a problem

        fct_values_Y = np.array([obj_fun(elm) for elm in sample_points_X]).squeeze()
        # todo could be accelerated by removing the for loop

        return self._get_coeffs_lin_reg(sample_points_X, fct_values_Y)

    # Instantiation of methods of parent class
    # ========================================
    def sketch_of_solution(self, sol=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).

        In: a solution P_theta, either None for the current sol, either a tuple (alpha,Theta) where
            - alpha is a (n_atoms,)-numpy array containing the weights
            - Theta is a (n_atoms,)
        Out: sketch_of_solution: (m,)-array containing the sketch
        """
        if sol is None:
            # Use the current solution
            (_alpha, _Atoms) = (self.alpha, self.Atoms)
        else:
            (_alpha, _Theta) = sol
            _Atoms = self.compute_atoms_matrix(_Theta)
        return _Atoms @ _alpha

    def fit_once(self, random_restart=True, n_iterations=None):
        """
        If random_restart is True, constructs a new solution from scratch with CLOMPR, else fine-tune.
        """

        if random_restart:
            ## Main mode of operation

            # Initializations
            if n_iterations is None:
                n_iterations = 2 * self.K  # By default: CLOMP-*R* (repeat twice)
            self.initialize_empty_solution()
            self.residual = self.sketch_reweighted

            # Main loop
            for i_iteration in range(n_iterations):
                ## Step 1: find new atom theta most correlated with residual
                new_theta = self.randomly_initialize_new_atom()
                new_theta = self.maximize_atom_correlation(new_theta)

                ## Step 2: add it to the support
                self.add_atom(new_theta)

                ## Step 3: if necessary, hard-threshold to enforce sparsity
                if self.n_atoms > self.K:
                    beta = self.find_optimal_weights(normalize_atoms=True)
                    index_to_remove = np.argmin(beta)
                    self.remove_atom(index_to_remove)
                    # Shortcut: if the last added atom is removed, we can skip to next iter
                    if index_to_remove == self.K:
                        continue

                ## Step 4: project to find weights
                self.alpha = self.find_optimal_weights()

                ## Step 5: fine-tune
                self.minimize_cost_from_current_sol()

                # Cleanup
                self.update_atoms()  # The atoms have changed: we must re-compute their sketches matrix
                self.residual = self.sketch_reweighted - self.sketch_of_solution()

        # Final fine-tuning with increased optimization accuracy
        self.minimize_cost_from_current_sol(ftol=0.02 * self.step5_ftol)

        # Normalize weights to unit sum
        self.alpha /= np.sum(self.alpha)

        # Package into the solution attribute
        self.current_sol = (self.alpha, self.Theta)