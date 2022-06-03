"""
Contains the CLOMP class for mixture estimation using CLOMP algorithm.
"""
from typing import NoReturn, Literal

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


class CLOMP(SolverTorch):
    """
    Implementation of the CLOMP algorithm to fit the sketch of a mixture model to the sketch z of a distribution.

    CLOMP is an instance of the class :class:`pycle.compressive_learning.SolverTorch.SolverTorch`.

    This class can use gradient descent through `torch` to find the components of the mixture model
    or it can use derivative free optimization through the `pdfo` library.
    Derivative free optimization is slower and it doesn't support too high dimension (>100) but doesn't need the
    feature map to be derivable.

    To create a subclass inheriting from CLOMP algorithm, some methods must be overriden:

    - `randomly_initialize_several_mixture_components(self, int)` to define how to initialize a given number of mixture components.
    - `sketch_of_mixture_components(self, (KxD) tensor )` to define how to get the feature map of a single or K mixture components
    - `set_bounds_thetas(bounds)` to define the bounding box where to look for the mixture components.

    To have a better understanding, look at the code of the class :class:`pycle.compressive_learning.CLOMP_CKM.CLOMP_CKM`

    References
    ----------
    Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
    Information and Inference: A Journal of the IMA, 7(3), 447-508.
    https://arxiv.org/pdf/1606.02838.pdf
    """
    # cleaning make dynamic reference to SOlverTorch rst

    LST_OPT_METHODS_TORCH = ["adam", "lbfgs"]

    def __init__(self, *args, **kwargs):
        self.maxiter_inner_optimizations = None
        self.tol_inner_optimizations = None
        self.nb_iter_max_step_5 = None
        self.nb_iter_max_step_1 = None
        self.opt_method_step_1 = None
        self.opt_method_step_34 = None
        self.opt_method_step_5 = None
        self.lr_inner_optimizations = None

        super().__init__(*args, **kwargs)

        self.weight_lower_bound = 1e-9
        self.weight_upper_bound = 2

        self.count_iter_torch_find_optimal_weights = 0
        self.count_iter_torch_maximize_atom_correlation = 0
        self.count_iter_torch_minimize_cost_from_current_sol = 0

    def initialize_hyperparameters_optimization(self) -> None:
        """
        Transform optimization parameters in dct_optim_method_hyperparameters to actual attributes of the object.

        Default key values for the dct_optim_method_hyperparameters dictionnary are::

            {
                "maxiter_inner_optimizations": 15000,  # Max number of iterations for all torch optimizations
                "tol_inner_optimizations": 1e-9,  # Change tolerance before stopping iterating in all torch optimizations
                "nb_iter_max_step_5": 200, # Max number of iterations for PDFO in step 5 (global finetuning)
                "nb_iter_max_step_1": 200, # Max number of iterations for PDFO in step 1 (find new cluster center)
                "opt_method_step_1": "lbfgs", # Default optimization algorithm for step 1 (find new cluster center)
                "opt_method_step_34": "nnls", # Default optimization algorithm for step 3 and 4 (find best mixture weights)
                "opt_method_step_5": "lbfgs", # Default optimization algorithm for step 5 (global finetuning)
                "lr_inner_optimizations": 1  # Start learning rate for torch optimizations with Adam.
            }

        """
        self.maxiter_inner_optimizations = self.dct_optim_method_hyperparameters.get("maxiter_inner_optimizations", 15000)
        self.tol_inner_optimizations = self.dct_optim_method_hyperparameters.get("tol_inner_optimizations", 1e-9)
        self.nb_iter_max_step_5 = self.dct_optim_method_hyperparameters.get("nb_iter_max_step_5", 200)
        self.nb_iter_max_step_1 = self.dct_optim_method_hyperparameters.get("nb_iter_max_step_1", 200)
        self.opt_method_step_1 = self.dct_optim_method_hyperparameters.get("opt_method_step_1", "lbfgs")
        self.opt_method_step_34 = self.dct_optim_method_hyperparameters.get("opt_method_step_34", "nnls")
        self.opt_method_step_5 = self.dct_optim_method_hyperparameters.get("opt_method_step_5", "lbfgs")
        self.lr_inner_optimizations = self.dct_optim_method_hyperparameters.get("lr_inner_optimizations", 1)

    @abstractmethod
    def projection_step(self, thetas):
        """
        Project mixture component parameters vector theta (or a set of thetas) on the constraint specifed
        by self.centroid_project of class `Projector`.

        The modification is made in place.

        Parameters
        ----------
        thetas
            (D,) or (current_size_mixture, D)-shaped tensor containing the parameters vector to project.
        """
        raise NotImplementedError

    def sketch_of_solution(self, alphas, thetas=None, phi_thetas=None):
        """
        Returns the sketch of the solution, A_Phi(thetas, alphas) = sum_k^K {alpha_k \* phi_theta_k}.

        Parameters
        ----------
        alphas
            (current_size_mixture,) shaped tensor of the weights of the mixture in the solution.
        thetas
            (current_size_mixture, D) shaped tensor containing the component parameters of the solution.
        phi_thetas
            (M, current_size_mixture) shaped tensor of each component sketched. If None, then the sketch will be computed
            from alphas and thetas.

        Returns
        -------
            (M,)-shaped tensor containing the sketch of the mixture
        """
        assert thetas is not None or phi_thetas is not None

        if phi_thetas is None:
            phi_thetas = torch.transpose(self.sketch_of_mixture_components(thetas), 0, 1)
        return torch.matmul(phi_thetas, alphas.to(self.comp_dtype))

    def add_atom(self, new_theta) -> NoReturn:
        """
        Adding a new theta and the corresponding new phi_theta to the CLOMP object.

        This will be used in each iteration when the new theta has been found (end of step 1 of the algorithm).

        Parameters
        ----------
        new_theta:
            (D, )- shaped tensor containing a new mixture component to add to the solution.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        """
        self.current_size_mixture += 1
        self.thetas = torch.cat((self.thetas, torch.unsqueeze(new_theta, 0)), dim=0)
        sketch_atom = self.sketch_of_mixture_components(new_theta)
        self.phi_thetas = torch.cat((self.phi_thetas, torch.unsqueeze(sketch_atom, 1)), dim=1)

    def remove_one_component(self, index_to_remove) -> NoReturn:
        """
        Remove a theta and the corresponding phi_theta.

        Removing a component should happen during step 3 of the algorithm.

        Parameters
        ----------
        index_to_remove:
            The index of the component to remove. The one with the smallest coefficient in the mixture.

        References
        ----------
        Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf
        """
        self.current_size_mixture -= 1
        self.thetas = torch.cat((self.thetas[:index_to_remove], self.thetas[index_to_remove+1:]), dim=0)
        self.phi_thetas = torch.cat((self.phi_thetas[:, :index_to_remove], self.phi_thetas[:, index_to_remove + 1:]),
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
        sketch_of_atom = self.sketch_of_mixture_components(theta)
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_phi_theta_norm:
            norm_atom = torch.tensor(self.minimum_phi_theta_norm).to(self.device)

        # note the "minus 1" that transforms the problem into a minimization problem
        result = -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, self.residual))
        if self.store_objective_values:
            ObjectiveValuesStorage().add(float(result), "loss_atom_correlation")
        return result

    def find_optimal_weights(self, normalize_phi_thetas=False, prefix="") -> torch.Tensor:
        """
        Returns the optimal wheights for the current mixture components 
        by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses scipy.optimize.nnls or torch depending on the parameters of CLOMP object.

        Parameters
        ----------
        normalize_phi_thetas
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

        init_alphas = torch.zeros(self.current_size_mixture, device=self.device)

        all_atoms = self.phi_thetas

        if self.opt_method_step_34 == "nnls":
            return self._find_optimal_weights_nnls(all_atoms, normalize_phi_thetas=normalize_phi_thetas)
        elif self.opt_method_step_34 in self.LST_OPT_METHODS_TORCH:
            return self._find_optimal_weights_torch(init_alphas, all_atoms, prefix, normalize_phi_thetas=normalize_phi_thetas)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_34}")

    def _find_optimal_weights_nnls(self, phi_thetas, normalize_phi_thetas=False) -> torch.Tensor:
        """
        Returns the optimal weights for the input phi_thetas by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses scipy.optimize.nnls procedure to solve the problem. This means that
        the input tensor and the residual have to be cast to numpy object, hence inducing some latency
        if the tensors were stored on GPU.

        Parameters
        ----------
        phi_thetas
            (M, current_size_mixture)-shaped tensor containing the sketch of all components in the mixture.
        normalize_phi_thetas
            Tells to normalize the sketch of the components before fitting the weights.
            This is usefull to recover the weights illustrating the importance of each component in the mixture.

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
        phi_thetas = phi_thetas.cpu().numpy()

        # Stack real and imaginary parts if necessary
        if np.any(np.iscomplex(phi_thetas)):  # True if complex sketch output
            _A = np.r_[phi_thetas.real, phi_thetas.imag]
            _z = np.r_[self.sketch_reweighted.real.cpu().numpy(), self.sketch_reweighted.imag.cpu().numpy()]
        else:
            _A = phi_thetas
            _z = self.sketch_reweighted.cpu().numpy()

        if normalize_phi_thetas:
            norms = np.linalg.norm(phi_thetas, axis=0)
            norm_too_small = np.where(norms < self.minimum_phi_theta_norm)[0]
            if norm_too_small.size > 0:  # Avoid division by zero
                logger.debug(f'norm of some atoms is too small (min. {norms.min()}), changed to {self.minimum_phi_theta_norm}.')
                norms[norm_too_small] = self.minimum_phi_theta_norm
            _A = _A / norms

        # Use non-negative least squares to find optimal weights
        (_alpha, _) = scipy.optimize.nnls(_A, _z)

        return torch.from_numpy(_alpha).to(self.device)

    def _find_optimal_weights_torch(self, init_alphas, phi_thetas, prefix, normalize_phi_thetas=False) -> torch.Tensor:
        """
        Returns the optimal wheights for the input phi_thetas by solving the Non-Negative Least Square problem.

        This correspond to the third and fourth subproblem of the CLOMPR algorithm.

        This function uses torch to solve the problem.

        Parameters
        ----------
        init_alphas
            (current_size_mixture,)-tensor corresponding to the initial weights used for the optimization.
        phi_thetas
            (M, current_size_mixture)-shaped tensor containing the sketch of all components in the solution.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        normalize_phi_thetas
            Tells to normalize the phi_thetas before fitting the weights.
            This is usefull to recover weight illustrating the importance of each component in the mixture.

        References
        ----------
        - Keriven, N., Bourrier, A., Gribonval, R., & Pérez, P. (2018). Sketching for large-scale learning of mixture models.
        Information and Inference: A Journal of the IMA, 7(3), 447-508.
        https://arxiv.org/pdf/1606.02838.pdf

        Returns
        -------
            (k,) shaped tensor of the weights of the least square solution
        """
        if normalize_phi_thetas:
            phi_thetas = f.normalize(self.phi_thetas, dim=1, eps=self.minimum_phi_theta_norm)
        else:
            phi_thetas = self.phi_thetas

        log_alphas = torch.nn.Parameter(init_alphas, requires_grad=True)
        optimizer = self._initialize_optimizer(self.opt_method_step_34, [log_alphas])

        def closure():
            optimizer.zero_grad()

            loss = self.loss_global(phi_thetas=phi_thetas, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.store_objective_values:
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

    def loss_global(self, alphas, all_thetas=None, phi_thetas=None):
        """
        Objective function of the global optimization problem: fitting the moments to the sketch of the mixture.

        Parameters
        ----------
        alphas
            (current_size_mixture,) shaped tensor of the weights of the mixture in the solution.
        all_thetas
            (current_size_mixture, D) shaped tensor containing the centers of the solution.
        phi_thetas
            (M, current_size_mixture) shaped tensor of each center sketched. If None, then the sketch will be computed from alphas
            and thetas.

        Returns
        -------
            The value of the objective evaluated at the provided parameters.
        """
        assert all_thetas is not None or phi_thetas is not None, "Thetas and Atoms must not be both None"
        sketch_solution = self.sketch_of_solution(alphas, thetas=all_thetas, phi_thetas=phi_thetas)
        loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution) ** 2
        return loss

    def _stack_sol(self, alphas: np.ndarray = None, thetas: np.ndarray = None) -> np.ndarray:
        """
        Stacks *all* the atoms and their weights into one vector.

        Note that this only work for numpy.ndarray.

        This is useful for optimization function taking only one vector as input.

        Parameters
        ----------
        alphas:
            (current_size_mixture,) shaped ndarray of the weights of the mixture in the solution.
        thetas:
            (current_size_mixture, D) shaped ndarray containing the centers of the solution.

        Returns
        -------

        (current_size_mixture + (current_size_mixture*D), ) shaped ndarray of alls alphas and Theta flattened and stacked together.
        """
        if (thetas is not None) and (alphas is not None):
            _Theta, _alpha = thetas, alphas
        else:
            _Theta, _alpha = self.thetas, self.alphas

        return np.r_[_Theta.reshape(-1), _alpha]

    def _destack_sol(self, p):
        """
        Reverse operation of `self._stack_sol`. Get back the parameters for 1D vector.

        Note that this only work for numpy.ndarray.

        This is useful for optimization function taking only one vector as input.

        Parameters
        ----------
        p
            (current_size_mixture + (current_size_mixture*D), ) shaped ndarray of alls alphas and thetas
            flattened and stacked together.

        Returns
        -------
            (alphas, thetas) tuple:
                - alphas: (current_size_mixture,) shaped ndarray of the weights of the mixture in the solution.
                - thetas: (current_size_mixture, D) shaped ndarray containing the parameters of the solution components.
        """
        assert p.shape[-1] == self.current_size_mixture * (self.thetas_dimension_D + 1)
        if p.ndim == 1 or p.shape[0] == 1:
            p = p.squeeze()
            thetas = p[:self.thetas_dimension_D * self.current_size_mixture].reshape(self.current_size_mixture, self.thetas_dimension_D)
            alphas = p[-self.current_size_mixture:].reshape(self.current_size_mixture)
        else:
            # todo fix?
            raise NotImplementedError(f"Impossible to destack p of shape {p.shape}.")
            # thetas = p[:, :self.d_theta * self.current_size_mixture].reshape(-1, self.current_size_mixture,
            # self.d_theta)
            # alphas = p[:, -self.current_size_mixture:].reshape(-1, self.current_size_mixture)
        return alphas, thetas

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
        all_thetas = self.thetas

        if self.opt_method_step_5 == "pdfo":

            # log_alphas = torch.log(self.alphas)
            # return self._minimize_cost_from_current_sol_torch(log_alphas, thetas, prefix)
            return self._minimize_cost_from_current_sol_pdfo(self.alphas, all_thetas, prefix)
        elif self.opt_method_step_5 in self.LST_OPT_METHODS_TORCH:
            log_alphas = torch.log(self.alphas)
            return self._minimize_cost_from_current_sol_torch(log_alphas, all_thetas, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_5}")

    def _minimize_cost_from_current_sol_pdfo(self, alphas, thetas, prefix):
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        The method doesn't return anything because the values are updated in place.

        Step 5 in CLOMP-R algorithm.

        This method uses pdfo subroutine for derivative free optimization which involves conversion of parameters
        to numpy.ndarray. Also, because the subroutine only takes vectors as input,
        it needs the solution to be stacked together as a single vector.

        Parameters
        ----------
        alphas:
            (current_size_mixture,) shaped tensor of the weights of the mixture in the solution.
        thetas:
            (current_size_mixture, D) shaped tensor containing the parameters of the solution components.
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
            if self.store_objective_values:
                ObjectiveValuesStorage().add(float(result), f"minimize_cost_from_current_sol_pdfo/{prefix}")
            return result

        stacked_x_init = self._stack_sol(alphas=alphas.cpu().numpy(), thetas=thetas.cpu().numpy())
        bounds_Theta_alpha = self.bounds_atom * self.current_size_mixture + [[self.weight_lower_bound, self.weight_upper_bound]] * self.current_size_mixture
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

        self.thetas = torch.Tensor(_all_thetas).to(self.real_dtype).to(self.device)
        self.alphas = torch.Tensor(_alphas).to(self.real_dtype).to(self.device)

    def _minimize_cost_from_current_sol_torch(self, log_alphas, thetas, prefix) -> NoReturn:
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        The method doesn't return anything because the values are updated in place.

        Step 5 in CLOMP-R algorithm.

        This method uses torch optimization.

        Parameters
        ----------
        log_alphas:
            (current_size_mixture,) shaped tensor of the logs of the weights of the mixture in the solution. We take the log
            because it allows to constraint the weights to be positive (by taking the exp)
        thetas:
            (current_size_mixture, D) shaped tensor containing the parameters of the solution components.
        prefix
            Prefix the identifier of the list of objective values, if they are stored.
        """
        # Parameters, optimizer
        log_alphas = log_alphas.requires_grad_()
        thetas = thetas.requires_grad_()
        params = [log_alphas, thetas]

        optimizer = self._initialize_optimizer(self.opt_method_step_5, params)

        def closure():
            optimizer.zero_grad()

            loss = self.loss_global(all_thetas=thetas, alphas=torch.exp(log_alphas).to(self.real_dtype))

            if self.tensorboard:
                self.writer.add_scalar(self.path_template_tensorboard_writer.format('step5'), loss.item(), iteration)
            if self.store_objective_values:
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
                self.projection_step(thetas)

            # Tracking loss
            if iteration != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / previous_loss

                if relative_loss_diff.item() < self.tol_inner_optimizations:
                    break

            previous_loss = torch.clone(loss)

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

        self.thetas = thetas.detach().to(self.device)
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
        self.update_current_solution_and_cost(new_current_solution=(self.thetas, self.alphas))

    def final_fine_tuning(self) -> NoReturn:
        """
        Minimise the global cost by tuning the whole solution (weights and centers).

        This is used as the last finetuning step in CLOMP.
        """
        logger.info(f'Final fine-tuning...')
        self.minimize_cost_from_current_sol(prefix="final")
        # self.projection_step(self.thetas) # this is useless given the projection was made in the last method
        logger.debug(torch.norm(self.thetas[:, :self.phi.d], dim=1))
        self.update_current_solution_and_cost(new_current_solution=(self.thetas, self.alphas))

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
        Step 1 in CLOMP-R algorithm. Find the theta giving the most correlated atom to the residual.

        Optimization can use pdfo or torch.

        Parameters
        ----------
        prefix
            Prefix the identifier of the list of objective values, if they are stored.

        Returns
        -------
            (D,)-shaped tensor corresponding to the new center.
        """
        new_theta = self.randomly_initialize_several_mixture_components(1).squeeze()

        if self.opt_method_step_1 == "pdfo":
            return self._maximize_atom_correlation_pdfo(new_theta)
            # return self._maximize_atom_correlation_torch(new_theta, prefix)
        elif self.opt_method_step_1 in self.LST_OPT_METHODS_TORCH:
            return self._maximize_atom_correlation_torch(new_theta, prefix)
        else:
            raise ValueError(f"Unkown optimization method: {self.opt_method_step_1}")

    def _maximize_atom_correlation_pdfo(self, new_theta: torch.Tensor) -> torch.Tensor:
        """
        Step 1 in CLOMP-R algorithm. Find the theta giving the most correlated atom to the residual.

        This method uses pdfo subroutine for derivative free optimization which involves conversion of parameters
        to numpy.ndarray.

        Parameters
        ----------
        new_theta
            (D,)-shaped tensor corresponding to the initial value for the mixture component parameter.

        References
        ----------
        - pdfo software: https://www.pdfo.net/index.html

        Returns
        -------
            (D,)-shaped tensor corresponding to the parameter of the new component.
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

            if self.store_objective_values:
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
        n_iterations = 2 * self.size_mixture_K

        for i_iter in range(n_iterations):
            logger.debug(f'Iteration {i_iter + 1} / {n_iterations}')
            # Step 1: find new atom theta most correlated with residual
            since = time.time()
            new_theta = self.maximize_atom_correlation(prefix=str(i_iter))
            logger.debug(f'Time for step 1: {time.time() - since}')

            # Step 2: add it to the support
            self.add_atom(new_theta)

            # Step 3: if necessary, hard-threshold to enforce sparsity
            if self.current_size_mixture > self.size_mixture_K:
                since = time.time()
                # atoms must be normalized so that the weights in beta reflect their importance. See Reference.
                beta = self.find_optimal_weights(normalize_phi_thetas=True, prefix=f"{i_iter}a")
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_one_component(index_to_remove)
                logger.debug(f'Time for step 3: {time.time() - since}')
                if index_to_remove == self.size_mixture_K:
                    logger.debug(f"Removed atom is the last one added. Solution is not updated.")
                    continue

            # Step 4 and 5
            self.do_step_4_5(prefix=str(i_iter))

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
        self.alphas /= torch.sum(self.alphas)
        self.update_current_solution_and_cost(new_current_solution=(self.thetas, self.alphas))

