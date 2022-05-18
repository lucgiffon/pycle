from typing import Any, NoReturn, Literal

import time
from abc import abstractmethod
import torch
import scipy
from pdfo import pdfo
from loguru import logger
import torch.nn.functional as f
from pycle.compressive_learning.SolverTorch import SolverTorch
from pycle.utils.intermediate_storage import ObjectiveValuesStorage
import numpy as np


# cleaning check documentation and make it numpy like
class CLOMP(SolverTorch):
    """
    Template for a compressive learning solver, using torch implementation, to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.

    Some size of tensors to keep in mind:
        - alphas: (n_atoms,)-tensor, weights of the mixture elements
        - all_thetas:  (n_atoms,D)-tensor, all the found parameters in matrix form. Each parameter is a "center".
        - all_atoms: (M,n_atoms)-tensor, the sketch of each theta (m is sketch size).

    The "solution" is the pair alphas, all_thetas, that is: the weights of the mixture with their corresponding centers.
    """

    LST_OPT_METHODS_TORCH = ["adam", "lbfgs"]

    def __init__(self, *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.weight_lower_bound = 1e-9
        self.weight_upper_bound = 2

        self.count_iter_torch_find_optimal_weights = 0
        self.count_iter_torch_maximize_atom_correlation = 0
        self.count_iter_torch_minimize_cost_from_current_sol = 0

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

        Parameters
        ----------
        alphas:
            (n_atoms,) shaped tensor of the weights of the mixture in the solution.
        all_thetas:
            (n_atoms, D) shaped tensor containing the centers of the solution.
        all_atoms
            (M, n_atoms) shaped tensor of each center sketched. If None, then the sketch will be computed from alphas and thetas.

        Returns
        -------
            (M,)-shaped tensor containing the sketch of the mixture
        """
        assert all_thetas is not None or all_atoms is not None

        if all_atoms is None:
            all_atoms = torch.transpose(self.sketch_of_atoms(all_thetas), 0, 1)
        return torch.matmul(all_atoms, alphas.to(self.comp_dtype))

    def add_atom(self, new_theta) -> NoReturn:
        """
        Adding a new theta (new center) and the corresponding new atom to the object.

        This will be used in each iteration when the new atom has been found (end of step 1 of the algorithm).

        Parameters
        ----------
        new_theta:
            (D, )- shaped tensor containing a new center to add to the solution.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        """
        self.n_atoms += 1
        self.all_thetas = torch.cat((self.all_thetas, torch.unsqueeze(new_theta, 0)), dim=0)
        sketch_atom = self.sketch_of_atoms(new_theta)
        self.all_atoms = torch.cat((self.all_atoms, torch.unsqueeze(sketch_atom, 1)), dim=1)

    def remove_one_atom(self, index_to_remove) -> NoReturn:
        """
        Remove a theta (a center) and the corresponding atom.

        Removing an atom should happen during step 3 of the algorithm.

        Parameters
        ----------
        index_to_remove:
            The index of the atom/center to remove. The one with the smallest coefficient in the mixture.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf
        """
        self.n_atoms -= 1
        self.all_thetas = torch.cat((self.all_thetas[:index_to_remove], self.all_thetas[index_to_remove+1:]), dim=0)
        self.all_atoms = torch.cat((self.all_atoms[:, :index_to_remove], self.all_atoms[:, index_to_remove + 1:]),
                                   dim=1)

    def loss_atom_correlation(self, theta):
        """
        Compute the correlation between sketch of the input theta and the residual of the current solution.

        This is the objective function of the step 1 of the algorithm.

        Parameters
        ----------
        theta
            (D,) -shaped tensor containg the current location where to estimate the correlation with the residual.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        Returns
        -------
            The value of the objective function evaluated at theta.
        """
        sketch_of_atom = self.sketch_of_atoms(theta)
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_atom_norm:
            norm_atom = torch.tensor(self.minimum_atom_norm).to(self.device)

        # note the "minus 1" that transforms the problem into a minimization problem
        result = -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, self.residual))
        if self.show_curves:
            ObjectiveValuesStorage().add(float(result), "loss_atom_correlation")
        return result

    def find_optimal_weights(self, normalize_atoms=False, prefix="") -> torch.Tensor:
        """
        Returns the optimal wheights for the input atoms by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses scipy.optimize.nnls or torch depending on the parameters of CLOMP object.

        Parameters
        ----------
        normalize_atoms
            Tells to normalize the atoms before fitting the weights. This is usefull to recover weight illustrating the
            importance of each atom in the mixture.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        Returns
        -------
            (k,) shaped tensor of the weights of the least square solution
        """

        init_alphas = torch.zeros(self.n_atoms, device=self.device)

        all_atoms = self.all_atoms

        if self.opt_method_step_34 == "nnls":
            return self._find_optimal_weights_nnls(all_atoms, normalize_atoms=normalize_atoms)
        elif self.opt_method_step_34 in self.LST_OPT_METHODS_TORCH:
            return self._find_optimal_weights_torch(init_alphas, all_atoms, prefix, normalize_atoms=normalize_atoms)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_34}")

    def _find_optimal_weights_nnls(self, all_atoms, normalize_atoms=False) -> torch.Tensor:
        """
        Returns the optimal wheights for the input atoms by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses scipy.optimize.nnls procedure to solve the problem. This means that
        the input tensor and the residual have to be cast to numpy object, hence inducing some latency
        if the tensor were stored on GPU.

        Parameters
        ----------
        all_atoms
            (M, n_atoms)-shaped tensor containing the sketch of all centers in the solution.
        normalize_atoms
            Tells to normalize the atoms before fitting the weights. This is usefull to recover weight illustrating the
            importance of each atom in the mixture.

        References
        ----------
        - Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf
        - scipy.optimize.nnls: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

        Returns
        -------
            (k,) shaped tensor of the weights of the least square solution
        """
        # scipy.optimize.nnls uses numpy arrays as input
        all_atoms = all_atoms.cpu().numpy()

        # Stack real and imaginary parts if necessary
        if np.any(np.iscomplex(all_atoms)):  # True if complex sketch output
            _A = np.r_[all_atoms.real, all_atoms.imag]
            _z = np.r_[self.sketch_reweighted.real.cpu().numpy(), self.sketch_reweighted.imag.cpu().numpy()]
        else:
            _A = all_atoms
            _z = self.sketch_reweighted.cpu().numpy()

        if normalize_atoms:
            norms = np.linalg.norm(all_atoms, axis=0)
            norm_too_small = np.where(norms < self.minimum_atom_norm)[0]
            if norm_too_small.size > 0:  # Avoid division by zero
                logger.debug(f'norm of some atoms is too small (min. {norms.min()}), changed to {self.minimum_atom_norm}.')
                norms[norm_too_small] = self.minimum_atom_norm
            _A = _A / norms

        # Use non-negative least squares to find optimal weights
        (_alpha, _) = scipy.optimize.nnls(_A, _z)

        return torch.from_numpy(_alpha).to(self.device)

    def _find_optimal_weights_torch(self, init_alphas, all_atoms, prefix, normalize_atoms=False) -> torch.Tensor:
        """
        Returns the optimal wheights for the input atoms by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses torch to solve the problem.

        Parameters
        ----------
        init_alphas
            (n_atoms,)-tensor corresponding to the initial weights used for the optimization.
        all_atoms
            (M, n_atoms)-shaped tensor containing the sketch of all centers in the solution.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        normalize_atoms
            Tells to normalize the atoms before fitting the weights. This is usefull to recover weight illustrating the
            importance of each atom in the mixture.

        References
        ----------
        - Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        Returns
        -------
            (k,) shaped tensor of the weights of the least square solution
        """
        if normalize_atoms:
            all_atoms = f.normalize(self.all_atoms, dim=1, eps=self.minimum_atom_norm)
        else:
            all_atoms = self.all_atoms

        log_alphas = torch.nn.Parameter(init_alphas, requires_grad=True)
        optimizer = self._initialize_optimizer(self.opt_method_step_34, [log_alphas])

        def closure():
            optimizer.zero_grad()

            loss = self.loss_global(all_atoms=all_atoms, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), "{}/find_optimal_weights".format(prefix))
            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format('step3-4'), loss.item(), i)

            loss.backward()
            return loss

        for i in range(self.maxiter_inner_optimizations):
            self.count_iter_torch_find_optimal_weights += 1
            if self.opt_method_step_34 == "lbfgs":
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

        # this exp trick allows to keep the alphas positive by design
        alphas = torch.exp(log_alphas)
        normalized_alphas = alphas / torch.sum(alphas)

        return normalized_alphas.detach()

    def loss_global(self, alphas, all_thetas=None, all_atoms=None):
        """
        Objective function of the global optimization problem: fitting the moments to the sketch of the mixture.

        Parameters
        ----------
        alphas:
            (n_atoms,) shaped tensor of the weights of the mixture in the solution.
        all_thetas:
            (n_atoms, D) shaped tensor containing the centers of the solution.
        all_atoms
            (M, n_atoms) shaped tensor of each center sketched. If None, then the sketch will be computed from alphas
            and thetas.

        Returns
        -------
            The value of the objective evaluated at the provided parameters.
        """
        assert all_thetas is not None or all_atoms is not None, "Thetas and Atoms must not be both None"
        sketch_solution = self.sketch_of_solution(alphas, all_thetas=all_thetas, all_atoms=all_atoms)
        loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution) ** 2
        return loss

    def _stack_sol(self, alpha: np.ndarray = None, Theta: np.ndarray = None) -> np.ndarray:
        """
        Stacks *all* the atoms and their weights into one vector.

        Note that this only work for numpy.ndarray.

        This is useful for optimization function taking only one vector as input.

        Parameters
        ----------
        alpha:
            (n_atoms,) shaped ndarray of the weights of the mixture in the solution.
        Theta:
            (n_atoms, D) shaped ndarray containing the centers of the solution.

        Returns
        -------

        (n_atoms + (n_atoms*D), ) shaped ndarray of alls alphas and Theta flattened and stacked together.
        """
        if (Theta is not None) and (alpha is not None):
            _Theta, _alpha = Theta, alpha
        else:
            _Theta, _alpha = self.all_thetas, self.alphas

        return np.r_[_Theta.reshape(-1), _alpha]

    def _destack_sol(self, p):
        """
        Reverse operation of `self._stack_sol`. Get back the parameters for 1D vector.

        Note that this only work for numpy.ndarray.

        This is useful for optimization function taking only one vector as input.

        Parameters
        ----------
        p
            (n_atoms + (n_atoms*D), ) shaped ndarray of alls alphas and Theta flattened and stacked together.

        Returns
        -------
            (alpha, Theta) tuple:
                - alpha: (n_atoms,) shaped ndarray of the weights of the mixture in the solution.
                - Theta: (n_atoms, D) shaped ndarray containing the centers of the solution.
        """
        assert p.shape[-1] == self.n_atoms * (self.d_theta + 1)
        if p.ndim == 1 or p.shape[0] == 1:
            p = p.squeeze()
            Theta = p[:self.d_theta * self.n_atoms].reshape(self.n_atoms, self.d_theta)
            alpha = p[-self.n_atoms:].reshape(self.n_atoms)
        else:
            # todo fix?
            raise NotImplementedError(f"Impossible to destack p of shape {p.shape}.")
            # Theta = p[:, :self.d_theta * self.n_atoms].reshape(-1, self.n_atoms, self.d_theta)
            # alpha = p[:, -self.n_atoms:].reshape(-1, self.n_atoms)
        return alpha, Theta

    def minimize_cost_from_current_sol(self, prefix="") -> NoReturn:
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        The method doesn't return anything because the values are updated in place.

        Step 5 in CLOMP-R algorithm.

        Parameters
        ----------
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        """
        # Parameters, optimizer
        all_thetas = self.all_thetas

        if self.opt_method_step_5 == "pdfo":

            # log_alphas = torch.log(self.alphas)
            # return self._minimize_cost_from_current_sol_torch(log_alphas, all_thetas, prefix)
            return self._minimize_cost_from_current_sol_pdfo(self.alphas, all_thetas, prefix)
        elif self.opt_method_step_5 in self.LST_OPT_METHODS_TORCH:
            log_alphas = torch.log(self.alphas)
            return self._minimize_cost_from_current_sol_torch(log_alphas, all_thetas, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_5}")

    def _minimize_cost_from_current_sol_pdfo(self, all_alphas, all_thetas, prefix):
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        The method doesn't return anything because the values are updated in place.

        Step 5 in CLOMP-R algorithm.

        This method uses pdfo subroutine for derivative free optimization which involves conversion of parameters
        to numpy.ndarray. Also, because the subroutine only takes vectors as input,
        it needs the solution to be stacked together as a single vector.

        Parameters
        ----------
        all_alphas:
            (n_atoms,) shaped tensor of the weights of the mixture in the solution.
        all_thetas:
            (n_atoms, D) shaped tensor containing the centers of the solution.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.

        References
        ----------
        - pdfo software: https://www.pdfo.net/index.html

        """
        def wrapped_loss_global(stacked_x):
            # the global loss doesn't take stacked parameters as input so it must be destacked first
            (_alpha, _Theta) = self._destack_sol(stacked_x)
            result = float(self.loss_global(all_thetas=torch.from_numpy(_Theta), alphas=torch.from_numpy(_alpha)))
            if self.show_curves:
                ObjectiveValuesStorage().add(float(result), f"minimize_cost_from_current_sol_pdfo/{prefix}")
            return result

        stacked_x_init = self._stack_sol(alpha=all_alphas.cpu().numpy(), Theta=all_thetas.cpu().numpy())
        bounds_Theta_alpha = self.bounds_atom * self.n_atoms + [[self.weight_lower_bound, self.weight_upper_bound]] * self.n_atoms
        # fct_fun_grad = self.get_global_cost
        fct_fun_grad = wrapped_loss_global
        sol = pdfo(fct_fun_grad,
                   x0=stacked_x_init,  # Start at current solution
                   bounds=bounds_Theta_alpha,
                   options={'maxfev': self.nb_iter_max_step_5 * stacked_x_init.size,
                            # 'rhoend': ftol
                            }
                   )

        (_alphas, _all_thetas) = self._destack_sol(sol.x)

        self.all_thetas = torch.Tensor(_all_thetas).to(self.real_dtype).to(self.device)
        self.alphas = torch.Tensor(_alphas).to(self.real_dtype).to(self.device)

    def _minimize_cost_from_current_sol_torch(self, log_alphas, all_thetas, prefix) -> NoReturn:
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        The method doesn't return anything because the values are updated in place.

        Step 5 in CLOMP-R algorithm.

        This method uses torch optimization.

        Parameters
        ----------
        log_alphas:
            (n_atoms,) shaped tensor of the logs of the weights of the mixture in the solution. We take the log
            because it allows to constraint the weights to be positive (by taking the exp)
        all_thetas:
            (n_atoms, D) shaped tensor containing the centers of the solution.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        """
        # Parameters, optimizer
        log_alphas = log_alphas.requires_grad_()
        all_thetas = all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]

        optimizer = self._initialize_optimizer(self.opt_method_step_5, params)

        def closure():
            optimizer.zero_grad()

            loss = self.loss_global(all_thetas=all_thetas, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format('step5'), loss.item(), iteration)
            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), f"{prefix}/minimize_cost_from_current_sol")

            loss.backward()
            return loss

        for iteration in range(self.maxiter_inner_optimizations):
            self.count_iter_torch_minimize_cost_from_current_sol += 1
            if self.opt_method_step_5 == "lbfgs":
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

        self.all_thetas = all_thetas.detach().to(self.device)
        self.alphas = torch.exp(log_alphas).detach().to(self.device)

    def do_step_4_5(self, prefix=""):
        """
        Do step 4 and 5 of CLOMP-R algorithm.

        Parameters
        ----------
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        """
        # Step 4: project to find weights
        since = time.time()
        self.alphas = self.find_optimal_weights(prefix=f"{prefix}b")
        logger.debug(f'Time for step 4: {time.time() - since}')
        # Step 5: fine-tune
        since = time.time()
        self.minimize_cost_from_current_sol(prefix=prefix)
        logger.debug(f'Time for step 5: {time.time() - since}')
        # The atoms have changed: we must re-compute their sketches matrix
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

    def final_fine_tuning(self) -> NoReturn:
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        This is used as the last finetuning step in CLOMP.
        """
        logger.info(f'Final fine-tuning...')
        self.minimize_cost_from_current_sol(prefix="final")
        # self.projection_step(self.all_thetas) # this is useless given the projection was made in the last method
        logger.debug(torch.norm(self.all_thetas[:, :self.phi.d], dim=1))
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

    def _initialize_optimizer(self, opt_method: Literal["adam", "lbfgs"], params: list):
        """
        Create an optimizer object according to the `opt_method`.

        Parameters
        ----------
        opt_method:
            Name of the optimization method.
        params:
            List of torch parameters to track and update by the optimizer.

        Returns
        -------
            The torch optimizer object.
        """
        if opt_method == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr_inner_optimizations)
        elif opt_method == "lbfgs":
            # learning rate is kept to 1 in that case because the rate is decided by strong_wolfe condition
            optimizer = torch.optim.LBFGS(params, max_iter=1, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Optimizer {opt_method} cannot be used for gradient descent. "
                             f"Choose one in {self.LST_OPT_METHODS_TORCH}")
        return optimizer

    def maximize_atom_correlation(self, prefix=""):
        """
        Step 1 in CLOMP-R algorithm. Find the center giving the most correlated atom to the residual.

        Optimization can use pdfo or torch.

        Parameters
        ----------
        prefix
            Prefix the identifier of the list of objective values, if they are stored.

        Returns
        -------
            (D,)-shaped tensor corresponding to the new center.
        """
        new_theta = self.randomly_initialize_several_atoms(1).squeeze()

        if self.opt_method_step_1 == "pdfo":
            return self._maximize_atom_correlation_pdfo(new_theta)
            # return self._maximize_atom_correlation_torch(new_theta, prefix)
        elif self.opt_method_step_1 in self.LST_OPT_METHODS_TORCH:
            return self._maximize_atom_correlation_torch(new_theta, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_1}")

    def _maximize_atom_correlation_pdfo(self, new_theta: torch.Tensor) -> torch.Tensor:
        """
        Step 1 in CLOMP-R algorithm. Find the center giving the most correlated atom to the residual.

        This method uses pdfo subroutine for derivative free optimization which involves conversion of parameters
        to numpy.ndarray.

        Parameters
        ----------
        new_theta
            (D,)-shaped tensor corresponding to the initial value for the cluster center.

        References
        ----------
        - pdfo software: https://www.pdfo.net/index.html

        Returns
        -------
            (D,)-shaped tensor corresponding to the new center.
        """
        # assert self.phi.device == torch.device("cpu")
        new_theta = new_theta.cpu().numpy()
        def fct_min_neg_atom_corr(x): return float(self.loss_atom_correlation(torch.from_numpy(x)))
        # fct_min_neg_atom_corr = self._get_residual_correlation_value
        sol = pdfo(fct_min_neg_atom_corr,
                   x0=new_theta,  # Start at current solution
                   bounds=self.bounds_atom,
                   options={'maxfev': self.nb_iter_max_step_1 * new_theta.size}
                   )

        return torch.Tensor(sol.x).to(self.real_dtype).to(self.device)

    def _maximize_atom_correlation_torch(self, new_theta, prefix):
        """
        Step 1 in CLOMP-R algorithm. Find the center giving the most correlated atom to the residual.

        This method uses torch for optimization.

        Parameters
        ----------
        new_theta
            (D,)-shaped tensor corresponding to the initial value for the cluster center.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.

        References
        ----------
        - pdfo software: https://www.pdfo.net/index.html

        Returns
        -------
            (D,)-shaped tensor corresponding to the new center.
        """
        params = [torch.nn.Parameter(new_theta, requires_grad=True)]

        optimizer = self._initialize_optimizer(self.opt_method_step_1, params)

        def closure():
            optimizer.zero_grad()
            loss = self.loss_atom_correlation(params[0])

            if self.show_curves:
                ObjectiveValuesStorage().add(float(loss), "{}/maximize_atom_correlation".format(prefix))
            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format("step1/{}".format(prefix)),
                                       loss.item(), i)

            loss.backward()
            return loss

        for i in range(self.maxiter_inner_optimizations):
            self.count_iter_torch_maximize_atom_correlation += 1
            if self.opt_method_step_1 == "lbfgs":
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

        return new_theta.data.detach()

    def fit_once(self):
        """
        CLOMP-R algorithm implementation.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf
        """
        # todo utiliser plutot une interface de type fit/transform
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
                # atoms must be normalized so that the weights in beta reflect their importance. See Reference.
                beta = self.find_optimal_weights(normalize_atoms=True, prefix=f"{i_iter}a")
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_one_atom(index_to_remove)
                logger.debug(f'Time for step 3: {time.time() - since}')
                if index_to_remove == self.nb_mixtures:
                    logger.debug(f"Removed atom is the last one added. Solution is not updated.")
                    continue

            # Step 4 and 5
            self.do_step_4_5(prefix=str(i_iter))

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
        self.alphas /= torch.sum(self.alphas)
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

