import time
from abc import abstractmethod
import torch
from dask.array.tests.test_stats import scipy
from pdfo import pdfo
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f
import socket
from datetime import datetime
from pycle.compressive_learning.torch.SolverTorch import SolverTorch
from pycle.utils.optim import ObjectiveValuesStorage
import numpy as np


class CLOMP(SolverTorch):
    """
    Template for a compressive learning solver, using torch implementation, to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.

    Some size of tensors to keep in mind:
        - alphas: (n_atoms,)-tensor, weigths of the mixture elements
        - all_thetas:  (n_atoms,d_atom)-tensor, all the found parameters in matrix form
        - all_atoms: (m,n_atoms)-tensor, the sketch of the found parameters (m is sketch size)
    """

    LST_OPT_METHODS_TORCH = ["adam", "lbfgs"]

    def __init__(self, phi, nb_mixtures, d_theta, bounds, sketch, sketch_weight=1., verbose=False,
                 path_template_tensorboard_writer="CLOMP/{}/loss/",  opt_method="adam", dct_opt_method=None,
                 **kwargs):
        """
        - phi: a FeatureMap object
        - sketch: tensor
        - sketch_weight: float, a re-scaling factor for the data sketch
        - verbose: bool
        """
        super().__init__(phi=phi, nb_mixtures=nb_mixtures, d_theta=d_theta, bounds=bounds, sketch=sketch, sketch_weight=sketch_weight, verbose=verbose, **kwargs)

        # Other minor params
        self.path_template_tensorboard_writer = path_template_tensorboard_writer

        # Attributes related to the optimization method used
        self.opt_method = opt_method
        self.dct_opt_method = dct_opt_method or dict()  # todo utiliser le dicitonnaire d'optim

        self.weight_lower_bound = 1e-9
        self.weight_upper_bound = 2

    # Abstract methods
    # ===============
    # Methods that have to be instantiated by child classes

    @abstractmethod
    def projection_step(self, theta):
        raise NotImplementedError

    # Generic methods
    # ===============
    # Methods that are general for all instances of this class
    # Instantiation of methods of parent class

    def sketch_of_solution(self, alphas, all_thetas=None, all_atoms=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).
        In: solution = (all_thetas, alphas)
            phi = sk.ComplexExpFeatureMap
            one_by_one = compute one atom by one atom in case atom computation does not fit in GPU
        Out: sketch_of_solution: (m,)-tensor containing the sketch
        """
        try:
            assert all_thetas is not None or all_atoms is not None
        except:
            pass
        if all_atoms is None:
            all_atoms = torch.transpose(self.sketch_of_atoms(all_thetas), 0, 1)
        return torch.matmul(all_atoms, alphas.to(self.comp_dtype))


    def add_atom(self, new_theta):
        """
        Adding a new atom.
        :param new_theta: tensor
        :return:
        """
        self.n_atoms += 1
        self.all_thetas = torch.cat((self.all_thetas, torch.unsqueeze(new_theta, 0)), dim=0)
        sketch_atom = self.sketch_of_atoms(new_theta)
        self.all_atoms = torch.cat((self.all_atoms, torch.unsqueeze(sketch_atom, 1)), dim=1)

    def remove_one_atom(self, index_to_remove):
        """
        Remove an atom.
        :param index_to_remove: int
        :return:
        """
        self.n_atoms -= 1
        self.all_thetas = torch.cat((self.all_thetas[:index_to_remove], self.all_thetas[index_to_remove+1:]), dim=0)
        self.all_atoms = torch.cat((self.all_atoms[:, :index_to_remove], self.all_atoms[:, index_to_remove + 1:]),
                                   dim=1)

    def loss_atom_correlation(self, theta):
        sketch_of_atom = self.sketch_of_atoms(theta)
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_atom_norm:
            norm_atom = torch.tensor(self.minimum_atom_norm)
        # note the "minus 1" that transforms the problem into a minimization problem
        return -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, self.residual))

    def find_optimal_weights(self, normalize_atoms=False, prefix=""):
        """
        Using the current atoms matrix, find the optimal weights for the sketch of the mixture to
        approximate well the true sketch. (Step 4)

        2nd optimization sub-problem in CLOMP
        """
        # todo make option to use nnls

        init_alphas = torch.zeros(self.n_atoms, device=self.device)
        if normalize_atoms:
            all_atoms = f.normalize(self.all_atoms, dim=1, eps=self.minimum_atom_norm)
        else:
            all_atoms = self.all_atoms

        if self.opt_method == "pdfo":
            # todo make a default for pdfo options for the opt method at each step of clomp
            return self._find_optimal_weights_torch(init_alphas, all_atoms, prefix)
        elif self.opt_method in self.LST_OPT_METHODS_TORCH:
            return self._find_optimal_weights_torch(init_alphas, all_atoms, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method}")

    def _find_optimal_weights_torch(self, init_alphas, all_atoms, prefix):
        """
        Using the current atoms matrix, find the optimal weights for the sketch of the mixture to
        approximate well the true sketch. (Step 4)

        2nd optimization sub-problem in CLOMP
        """
        log_alphas = torch.nn.Parameter(init_alphas, requires_grad=True)
        optimizer = self._initialize_optimizer([log_alphas])

        def closure():
            optimizer.zero_grad()

            # todo inconsistance entre les différentes fonctions pour calculer la loss
            loss = self.loss_global(all_atoms=all_atoms, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), "find_optimal_weights/{}".format(prefix))
            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format('step3-4'), loss.item(), i)

            loss.backward()
            return loss

        for i in range(self.maxiter_inner_optimizations):
            if self.opt_method == "lbfgs":
                loss = optimizer.step(closure)  # bfgs takes the loss computation function as argument at each step.
            else:
                loss = closure()
                optimizer.step()

            if i != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)

                if relative_loss_diff.item() <= self.tol_inner_optimizations:
                    break

            previous_loss = torch.clone(loss)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()
        if self.show_curves:
            ObjectiveValuesStorage().show()
            ObjectiveValuesStorage().clear()

        alphas = torch.exp(log_alphas)
        normalized_alphas = alphas / torch.sum(alphas)

        return normalized_alphas.detach()


    def loss_global(self, alphas, all_thetas=None, all_atoms=None):
        assert all_thetas is not None or all_atoms is not None
        sketch_solution = self.sketch_of_solution(alphas, all_thetas, all_atoms)
        loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution)
        return loss

    def _stack_sol(self, alpha=None, Theta=None):
        """Stacks *all* the atoms and their weights into one vector"""
        if (Theta is not None) and (alpha is not None):
            _Theta, _alpha = Theta, alpha
        else:
            _Theta, _alpha = self.all_thetas, self.alphas

        return np.r_[_Theta.reshape(-1), _alpha]

    def _destack_sol(self, p):
        assert p.shape[-1] == self.n_atoms * (self.d_theta + 1)
        if p.ndim == 1 or p.shape[0] == 1:
            p = p.squeeze()
            Theta = p[:self.d_theta * self.n_atoms].reshape(self.n_atoms, self.d_theta)
            alpha = p[-self.n_atoms:].reshape(self.n_atoms)
        else:
            # todo à corriger
            raise NotImplementedError
            Theta = p[:, :self.d_theta * self.n_atoms].reshape(-1, self.n_atoms, self.d_theta)
            alpha = p[:, -self.n_atoms:].reshape(-1, self.n_atoms)
        return alpha, Theta

    def minimize_cost_from_current_sol(self, prefix=""):
        """
        Step 5 in CLOMP-R algorithm. At the end of the method, update the parameters self.alphas and self.all_thetas.

        Third optimization subproblem.

        :param tol: float. Stopping criteria for optimization: when the relative difference in loss is less than tol.
        :param max_iter: int. Maximum number of iterations.
        :param tensorboard: set True to plot loss in Tensorboard.
        :return:
        """
        # Parameters, optimizer
        log_alphas = torch.log(self.alphas)
        all_thetas = self.all_thetas

        if self.opt_method == "pdfo":
            return self._minimize_cost_from_current_sol_pdfo(log_alphas, all_thetas, prefix)
        elif self.opt_method in self.LST_OPT_METHODS_TORCH:
            return self._minimize_cost_from_current_sol_torch(log_alphas, all_thetas, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method}")

    def _minimize_cost_from_current_sol_pdfo(self, log_alphas, all_thetas, prefix):
        def wrapped_loss_global(stacked_x):
            (_log_alpha, _Theta) = self._destack_sol(stacked_x)
            return float(self.loss_global(all_thetas=torch.from_numpy(_Theta), alphas=torch.from_numpy(_log_alpha)))

        stacked_x_init = self._stack_sol(alpha=log_alphas.numpy(), Theta=all_thetas.numpy())
        bounds_Theta_alpha = self.bounds_atom * self.n_atoms + [
            [self.weight_lower_bound, self.weight_upper_bound]] * self.n_atoms
        # fct_fun_grad = self.get_global_cost
        fct_fun_grad = wrapped_loss_global
        nb_iter_max = self.dct_opt_method.get("nb_iter_max_step_5", 5)
        sol = pdfo(fct_fun_grad,
                   x0=stacked_x_init,  # Start at current solution
                   bounds=bounds_Theta_alpha,
                   options={'maxfev': nb_iter_max * stacked_x_init.size,
                            # 'rhoend': ftol
                            }
                   )
        return torch.Tensor(sol.x).to(self.real_dtype)

    def _minimize_cost_from_current_sol_torch(self, log_alphas, all_thetas, prefix):
        # Parameters, optimizer
        log_alphas = log_alphas.requires_grad_()
        all_thetas = all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]

        optimizer = self._initialize_optimizer(params)

        def closure():
            optimizer.zero_grad()

            loss = self.loss_global(all_thetas=all_thetas, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format('step5'), loss.item(), iteration)
            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), f"minimize_cost_from_current_sol/{prefix}")

            loss.backward()
            return loss

        for iteration in range(self.maxiter_inner_optimizations):
            if self.opt_method == "lbfgs":
                loss = optimizer.step(closure)  # bfgs takes the loss computation function as argument at each step.
            else:
                loss = closure()
                optimizer.step()

            # Projection step
            with torch.no_grad():
                self.projection_step(all_thetas)

            # Tracking loss
            if iteration != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / previous_loss

                if relative_loss_diff.item() < self.tol_inner_optimizations:
                    break

            previous_loss = torch.clone(loss)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()
        if self.show_curves:
            ObjectiveValuesStorage().show()
            ObjectiveValuesStorage().clear()

        self.all_thetas = all_thetas.detach()
        self.alphas = torch.exp(log_alphas).detach()

    def do_step_4_5(self):
        """
        Do step 4 and 5 of CLOMP-R.
        :param tensorboard:
        :return:
        """
        # Step 4: project to find weights
        since = time.time()
        self.alphas = self.find_optimal_weights()
        logger.debug(f'Time for step 4: {time.time() - since}')
        # Step 5: fine-tune
        since = time.time()
        self.minimize_cost_from_current_sol()
        logger.debug(f'Time for step 5: {time.time() - since}')
        # The atoms have changed: we must re-compute their sketches matrix
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

    def final_fine_tuning(self):
        logger.info(f'Final fine-tuning...')
        self.minimize_cost_from_current_sol()
        # self.projection_step(self.all_thetas) # this is useless given the projection was made in the last method
        logger.debug(torch.norm(self.all_thetas[:, :self.phi.d], dim=1))
        self.alphas /= torch.sum(self.alphas)
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))
        # self.current_sol =

    def _initialize_optimizer(self, params):
        if self.opt_method == "adam" or self.opt_method == "pdfo":
            optimizer = torch.optim.Adam(params, lr=self.lr_inner_optimizations)
        elif self.opt_method == "lbfgs":
            optimizer = torch.optim.LBFGS(params, max_iter=1, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"unkown opt method: {self.opt_method}")
        return optimizer

    def _maximize_atom_correlation_pdfo(self, new_theta, prefix):
        assert self.phi.device == torch.device("cpu")
        new_theta = new_theta.numpy()
        fct_min_neg_atom_corr = lambda x: float(self.loss_atom_correlation(torch.from_numpy(x)))
        nb_iter_max = self.dct_opt_method.get("nb_iter_max_step_1", 100)
        sol = pdfo(fct_min_neg_atom_corr,
                   x0=new_theta,  # Start at current solution
                   bounds=self.bounds_atom,
                   options={'maxfev': nb_iter_max * new_theta.size}
                   )
        return torch.Tensor(sol.x).to(self.real_dtype)

    def _maximize_atom_correlation_torch(self, new_theta, prefix):
        params = [torch.nn.Parameter(new_theta, requires_grad=True)]

        optimizer = self._initialize_optimizer(params)

        def closure():
            optimizer.zero_grad()
            loss = self.loss_atom_correlation(params[0])

            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), "maximize_atom_correlation/{}".format(prefix))
            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format("step1/{}".format(prefix)),
                                       loss.item(), i)

            loss.backward()
            return loss

        for i in range(self.maxiter_inner_optimizations):
            if self.opt_method == "lbfgs":
                loss = optimizer.step(closure)  # bfgs takes the loss computation function as argument at each step.
            else:
                loss = closure()
                optimizer.step()

            # Projection step
            with torch.no_grad():
                self.projection_step(new_theta)

            if i != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if relative_loss_diff.item() <= self.tol_inner_optimizations:
                    break
            previous_loss = torch.clone(loss)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

        if self.show_curves:
            ObjectiveValuesStorage().show()
            ObjectiveValuesStorage().clear()

        return new_theta.data.detach()

    def maximize_atom_correlation(self, prefix=""):
        """
        Step 1 in CLOMP-R algorithm. Find most correlated atom. Torch optimization, using Adam.

        :param new_theta: torch tensor for atom
        :param tol: stopping criteria is to stop when the relative difference of loss between
        two consecutive iterations is less than tol.
        :param max_iter: max iterations number for optimization.
        :return: updated new_theta
        """
        new_theta = self.randomly_initialize_several_atoms(1).squeeze()

        if self.opt_method == "pdfo":
            return self._maximize_atom_correlation_pdfo(new_theta, prefix)
        elif self.opt_method in self.LST_OPT_METHODS_TORCH:
            return self._maximize_atom_correlation_torch(new_theta, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method}")

    def fit_once(self):
        """
        CLOMP-R algorithm implementation.
        If random_restart is True, constructs a new solution from scratch with CLOMP-R, else fine-tune.
        """
        n_iterations = 2 * self.nb_mixtures

        for i_iter in range(n_iterations):
            logger.debug(f'Iteration {i_iter + 1} / {n_iterations}')
            # Step 1: find new atom theta most correlated with residual
            since = time.time()
            new_theta = self.maximize_atom_correlation(prefix=str(i_iter))
            logger.debug(f'Time for step 1: {time.time() - since}')

            # Step 2: add it to the support
            self.add_atom(new_theta)

            # Step 3: if necessary, hard-threshold to enforce sparsity
            if self.n_atoms > self.nb_mixtures:
                since = time.time()
                beta = self.find_optimal_weights(normalize_atoms=True)
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_one_atom(index_to_remove)
                logger.debug(f'Time for step 3: {time.time() - since}')
                if index_to_remove == self.nb_mixtures:
                    continue

            # Step 4 and 5
            self.do_step_4_5()

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
