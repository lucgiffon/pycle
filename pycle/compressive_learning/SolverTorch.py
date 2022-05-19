from typing import NoReturn

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod

from pycle.sketching import FeatureMap


# cleaning make documentation and clean everything here
# cleaning make UML diagram of everything that is happening here
# 0.1 Generic solver (stores a sketch and a solution,
# can run multiple trials of a learning method to specify)
class SolverTorch(metaclass=ABCMeta):
    """
    Template for a compressive learning, mixture model estimation solver.

    It is used to fit the sketch of a mixture model to the sketch z of a distribution,
    that is to solve problems of the form:

        min_({alpha_k, theta_k}_{k=1}^K) || sketch_weight * z - A_Phi(sum_{k=1}^{K} {alpha_k * dirac_theta_k} ) ||_2.

    """
    def __init__(self, phi: FeatureMap, sketch_z, size_mixture_K, D_theta, bounds,
                 store_objective_values: bool = False,
                 tensorboard: bool = False, path_template_tensorboard_writer="CLOMP/{}/loss/",
                 dct_opt_method: [None, dict] = None, sketch_weight=1.):
        """
        Parameters
        ----------
        phi
            The feature map underlying the sketching operator.
        sketch_z
            The sketch of the distribution intended to be modeled by a mixture model.
        size_mixture_K
            The number K of components in the mixture.
        D_theta
            The dimension of each component in the mixture.
        bounds
            (2, D)- shaped tensor containing the lower bounds in position 0 and upper bounds in position 1.
        store_objective_values
            Tells to store the objective values in an :py:class:`.ObjectiveValuesStorage` singleton class.
        tensorboard
            Tells to track the optimization in a tensorboard pannel. The tensorboard results will be stored at path
            specified by `path_template_tensorboard_writer`.
        path_template_tensorboard_writer
            The path where to store the tensorboard results, if `tensorboard` argument is True.
        dct_opt_method
            Dictionary containing all the hyper-parameters relative to the optimization.
        sketch_weight
            The weight of the sketch. It can be a scalar if the whole sketch must be assigned a scaling
            or it can be a vector with the same dimension as the sketch, allowing to give different importance
            to each coefficient of the sketch.
        """
        # Attributes related to the optimization method used
        # cleaning verify that this dictionary usage is clean and well documented:
        #  the user must find what are the possible keys
        self.dct_opt_method = dct_opt_method or dict()  # todo utiliser le dicitonnaire d'optim
        self.initialize_parameters_optimization()

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch_z.device.type
        self.device = phi.device
        self.real_dtype = phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        # parameters regarding the tracking of the algorithm
        self.store_objective_values = store_objective_values
        self.tensorboard = tensorboard
        self.path_template_tensorboard_writer = path_template_tensorboard_writer
        if tensorboard:
            self.writer = SummaryWriter()

        # Encode feature map
        assert isinstance(phi, FeatureMap)
        self.phi = phi

        # Set other values
        self.nb_mixtures = size_mixture_K
        self.d_theta = D_theta
        self.n_atoms = 0

        # Encode sketch and sketch weight
        self.sketch = sketch_z
        self.sketch_weight = None
        self.sketch_reweighted = None
        self.update_sketch_and_weight(sketch_z, sketch_weight)

        self.alphas = None
        self.all_thetas = None
        self.all_atoms = None
        self.initialize_empty_solution()

        # Set bounds
        self.bounds = None
        self.bounds_atom = None
        self.set_bounds_atom(bounds)  # bounds for an atom

        # Encode current theta and cost value
        self.current_sol = None
        self.current_sol_cost = None
        self.residual = None
        self.update_current_sol_and_cost(None)

        self.counter_call_sketching_operator = 0

        self.minimum_atom_norm = 1e-15 * np.sqrt(self.d_theta)

    def initialize_parameters_optimization(self) -> None:
        """
        Transform optimization parameters in dct_opt_method to actual attributes of the object.
        Further tests could be done here, as the adequation between the optimization method used and the parameters provided.
        :return:
        """
        # cleaning this function should not be aware of what happens in child classes
        self.maxiter_inner_optimizations = self.dct_opt_method.get("maxiter_inner_optimizations", 15000)
        self.tol_inner_optimizations = self.dct_opt_method.get("tol_inner_optimizations", 1e-9)
        self.lr_inner_optimizations = self.dct_opt_method.get("lr_inner_optimizations", 1)
        self.beta_1 = self.dct_opt_method.get("beta_1", 0.9)
        self.beta_2 = self.dct_opt_method.get("beta_2", 0.99)
        self.nb_iter_max_step_5 = self.dct_opt_method.get("nb_iter_max_step_5", 200)
        self.nb_iter_max_step_1 = self.dct_opt_method.get("nb_iter_max_step_1", 200)

        self.opt_method_step_1 = self.dct_opt_method.get("opt_method_step_1", "lbfgs")
        self.opt_method_step_34 = self.dct_opt_method.get("opt_method_step_34", "nnls")
        self.opt_method_step_5 = self.dct_opt_method.get("opt_method_step_5", "lbfgs")

        self.lambda_l1 = self.dct_opt_method.get("lambda_l1", 0)

    def update_current_sol_and_cost(self, sol=None) -> NoReturn:
        """
        Updates the residual and cost to the current solution.
        If `sol` given, also updates the `current_sol` attribute.

        Parameters
        ----------
        sol
            (alphas, thetas) corresponding to the current weights and coefficients of the solution.
        """

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(all_thetas=self.current_sol[0],
                                                                             alphas=self.current_sol[1])
            self.current_sol_cost = torch.norm(self.residual) ** 2
        else:
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf

    def initialize_empty_solution(self) -> None:
        """
        This method prepares the attributes that will contain the solution of the compressive learning problem.

        Attributes pertaining to the solution are :

         - `n_atoms`, `alphas`, `all_thetas`, `all_atoms`, `residual` and `current_sol`.

        """
        self.n_atoms = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)
        self.all_atoms = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)
        self.current_sol = (self.all_thetas, self.alphas)  # Overwrite

    @abstractmethod
    def set_bounds_atom(self, bounds):
        """
        Set the bounds where the cluster centers can be found.

        These bounds can be used for initizalizing new cluster centers
        and for setting bounds to the optimization procedure.

        Parameters
        ----------
        bounds
            (2, D)- shaped tensor containing the lower bounds in position 0 and upper bounds in position 1.
        """
        pass

    @abstractmethod
    def sketch_of_atoms(self, thetas):
        """
        Computes and returns A_Phi(theta_k) for each cluster center in theta.

        D is the dimension of cluster center, M is the dimension of a sketch.

        Parameters
        ----------
        theta
            (D,) or (n_atoms,D)-shaped tensor containing cluster centers.

        Returns
        -------
            (M,) or (n_atoms, M)-shaped tensor constaining the M-dimensional feature maps of the cluster centers,
            e.g. the atoms.
        """
        raise NotImplementedError

    @abstractmethod
    def sketch_of_solution(self, alphas, all_thetas=None, all_atoms=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).

        Parameters
        ----------
        alphas:
            (n_atoms,) shaped tensor of the weights of the mixture in the solution.
        all_thetas:
            (n_atoms, D) shaped tensor containing the mixture components of the solution.
        all_atoms
            (M, n_atoms) shaped tensor of each component sketched separately.
            If None, then the sketch will be computed from alphas and thetas.

        Returns
        -------
            (M,)-shaped tensor containing the sketch of the mixture
        """
        raise NotImplementedError

    @abstractmethod
    def fit_once(self):
        """
        The optimization procedure to fit the sketch of the mixture to the sketch.
        """
        raise NotImplementedError

    @abstractmethod
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number mixture components.

        Parameters
        ----------
        nb_atoms
            The number of mixture components to initialize.

        Returns
        -------
            (nb_atoms, D) shaped tensor containing the mixture components parameters.
        """
        raise NotImplementedError

    # Generic methods
    # ===============
    def update_sketch_and_weight(self, sketch=None, sketch_weight=None) -> NoReturn:
        """
        Updates the residual and cost to the current solution. If sol given, also updates it.

        Parameters
        ----------
        sketch
            (M,) shaped tensor containing the new sketch to consider.
        sketch_weight
            (M,) shaped tensor or scalar containing the weights for the sketch coefficients.
        """
        if sketch is not None:
            self.sketch = sketch
        if sketch_weight is not None:
            assert isinstance(sketch_weight, float) or isinstance(sketch_weight, int)
            self.sketch_weight = sketch_weight
        # it is sketch_reweighted that should be used for optimizing.
        self.sketch_reweighted = self.sketch_weight * self.sketch

    def fit_several_times(self, n_repetitions=1, forget_current_sol=False):
        """
        Solves the problem `n_repetitions` times, keep and return the best solution.

        Parameters
        ----------
        n_repetitions
            Number of times the algorithm must be repeated.
        forget_current_sol

        Returns
        -------

        """
        # cleaning test this

        # Initialization
        if forget_current_sol:
            # start from scratch
            best_sol, best_sol_cost = None, np.inf
        else:
            # keep current solution as candidate
            best_sol, best_sol_cost = self.current_sol, self.current_sol_cost

        # Main loop, perform independent trials
        for i_repetition in range(n_repetitions):
            self.fit_once()
            self.update_current_sol_and_cost()

            if self.current_sol_cost < best_sol_cost:
                best_sol, best_sol_cost = self.current_sol, self.current_sol_cost

        # Set the current sol to the best one we found
        self.update_current_sol_and_cost(sol=best_sol)
        return self.current_sol
