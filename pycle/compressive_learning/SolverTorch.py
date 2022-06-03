"""
Contains the SolverTorch class for mixture estimation.
"""

from typing import NoReturn

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod

from pycle.sketching import FeatureMap


class SolverTorch(metaclass=ABCMeta):
    """
    Template for a compressive learning, mixture model estimation solver.

    It is used to fit the sketch of a mixture model to the sketch z of a distribution,
    that is to solve problems of the form:

        min_({alpha_k, theta_k}_{k=1}^K) || sketch_weight * z - sum_{k=1}^{K} {alpha_k * phi_theta_k} ||_2.

    Definitions:
        - The sketched component k of the mixture model is denoted phi_theta_k
        - The parameters of the component k of the mixture model is denoted theta_k
        - The dimension of each parameter vector is D
        - The size of the mixture is K, the number of components
        - The alphas are the weights of the mixture
        - The solution is the pair (alphas, thetas), that is, all the parameters of the mixture

    Some size of tensors to keep in mind:
        - alphas: (current_size_mixture,)-tensor, weights of the mixture elements
        - thetas:  (current_size_mixture,D)-tensor, all the found parameters in matrix form. Each parameter is a "center".
        - phi_thetas: (M,current_size_mixture)-tensor, the sketch of each theta (m is sketch size).
    """
    def __init__(self, phi: FeatureMap, sketch_z, size_mixture_K, thetas_dimension_D, bounds,
                 store_objective_values: bool = False,
                 tensorboard: bool = False, path_template_tensorboard_writer="CLOMP/{}/loss/",
                 dct_optim_method_hyperparameters: [None, dict] = None, sketch_weight=1.):
        """
        Constructor.

        Parameters
        ----------
        phi
            The feature map underlying the sketching operator.
        sketch_z
            The sketch of the distribution intended to be modeled by a mixture model.
        size_mixture_K
            The number K of components in the mixture.
        thetas_dimension_D
            The dimension of the parameter vector of each component in the mixture.
        bounds
            (2, D)- shaped tensor containing the lower bounds in position 0 and upper bounds in position 1.
        store_objective_values
            Tells to store the objective values in an :py:class:`.ObjectiveValuesStorage` singleton class.
        tensorboard
            Tells to track the optimization in a tensorboard pannel. The tensorboard results will be stored at path
            specified by `path_template_tensorboard_writer`.
        path_template_tensorboard_writer
            The path where to store the tensorboard results, if `tensorboard` argument is True.
        dct_optim_method_hyperparameters
            Dictionary containing all the hyper-parameters relative to the optimization.
        sketch_weight
            The weight of the sketch. It can be a scalar if the whole sketch must be assigned a scaling
            or it can be a vector with the same dimension as the sketch, allowing to give different importance
            to each coefficient of the sketch.


        Attributes
        ----------
        current_solution
            (alphas, thetas) the pair of mixture components weights and parameters:
                - alphas: (current_size_mixture,)-shaped tensor containing the mixture components weights
                - thetas: (current_size_mixture, D)-shaped tensor containing the mixture components parameters
        """
        # Attributes related to the optimization method used
        self.dct_optim_method_hyperparameters = dct_optim_method_hyperparameters or dict()
        self.initialize_hyperparameters_optimization()

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
        self.size_mixture_K = size_mixture_K
        self.thetas_dimension_D = thetas_dimension_D
        self.current_size_mixture = 0

        # Encode sketch and sketch weight
        self.sketch_z = sketch_z
        self.sketch_weight = None
        self.sketch_reweighted = None
        self.update_sketch_and_weight(sketch_z, sketch_weight)

        self.alphas = None
        self.thetas = None
        self.phi_thetas = None
        self.initialize_empty_solution()

        # Set bounds
        self.bounds = None
        self.bounds_atom = None
        self.set_bounds_thetas(bounds)  # bounds for an atom

        # Encode current theta and cost value
        self.current_solution = None
        self.current_sol_cost = None
        self.residual = None
        self.update_current_solution_and_cost(None)

        self.counter_call_sketching_operator = 0

        self.minimum_phi_theta_norm = 1e-15 * np.sqrt(self.thetas_dimension_D)

    @abstractmethod
    def initialize_hyperparameters_optimization(self) -> None:
        """
        Transform optimization hyperparameters in dct_optim_method_hyperparameters to actual attributes of the object.
        """
        # # CLOMP and CLHS legacy class
        # self.lr_inner_optimizations = self.dct_optim_method_hyperparameters.get("lr_inner_optimizations", 1)
        #
        # # CLOMP_DL legacy class
        # self.lambda_l1 = self.dct_optim_method_hyperparameters.get("lambda_l1", 0)
        #
        # # CLHS legacy class
        # self.beta_1 = self.dct_optim_method_hyperparameters.get("beta_1", 0.9)
        # self.beta_2 = self.dct_optim_method_hyperparameters.get("beta_2", 0.99)
        raise NotImplementedError

    @abstractmethod
    def set_bounds_thetas(self, bounds):
        """
        Set the bounds where the mixture component parameters can be found.

        These bounds can be used for initizalizing new mixture components
        and for setting bounds to the optimization procedure.

        Parameters
        ----------
        bounds
            (2, D)- shaped tensor containing the lower bounds in position 0 and upper bounds in position 1.
        """
        pass

    @abstractmethod
    def sketch_of_mixture_components(self, thetas):
        """
        Computes and returns phi_theta_k for each theta provided, that is, each mixture component parameter vector.

        D is the dimension of a mixture component parameter vector, M is the dimension of a sketch.

        Parameters
        ----------
        thetas
            (D,) or (current_size_mixture,D)-shaped tensor containing the.

        Returns
        -------
            (M,) or (current_size_mixture, M)-shaped tensor constaining the M-dimensional feature maps of
            the mixture components
        """
        raise NotImplementedError

    @abstractmethod
    def sketch_of_solution(self, alphas, thetas=None, phi_thetas=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).

        Parameters
        ----------
        alphas:
            (current_size_mixture,) shaped tensor of the weights of the mixture in the solution.
        thetas:
            (current_size_mixture, D) shaped tensor containing the mixture components parameters of the solution.
        phi_thetas
            (M, current_size_mixture) shaped tensor of each component sketched separately.
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
    def randomly_initialize_several_mixture_components(self, nb_mixture_components):
        """
        Define how to initialize a given number of mixture components.

        Parameters
        ----------
        nb_mixture_components
            The number of mixture components to initialize.

        Returns
        -------
            (nb_atoms, D) shaped tensor containing the mixture components parameters.
        """
        raise NotImplementedError

    def initialize_empty_solution(self) -> None:
        """
        This method prepares the attributes that will contain the solution of the compressive learning problem.

        Attributes pertaining to the solution are :

         - `current_size_mixture`, `alphas`, `thetas`, `phi_thetas`, `residual` and `current_solution`.

        """
        # cleaning unittest it
        self.current_size_mixture = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.thetas = torch.empty(0, self.thetas_dimension_D, dtype=self.real_dtype).to(self.device)
        # cleaning this is very weird that phi_thetas and thetas do not have the same orientation
        #  if there is a justification, find it and document it. Otherwise, just keep the same!
        self.phi_thetas = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)
        self.current_solution = (self.thetas, self.alphas)  # Overwrite

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
        # cleaning test it
        if sketch is not None:
            self.sketch_z = sketch
        if sketch_weight is not None:
            assert isinstance(sketch_weight, float) or isinstance(sketch_weight, int)
            self.sketch_weight = sketch_weight
        # it is sketch_reweighted that should be used for optimizing.
        self.sketch_reweighted = self.sketch_weight * self.sketch_z

    def update_current_solution_and_cost(self, new_current_solution=None) -> NoReturn:
        """
        Updates the residual and cost to the current solution.
        If `new_current_solution` given, also updates the `current_solution` attribute of the `SolverTorch` object.

        Parameters
        ----------
        new_current_solution
            (alphas, thetas) corresponding to the current weights and coefficients of the solution.
        """
        # cleaning unittest it
        # Update current sol if argument given
        if new_current_solution is not None:
            self.current_solution = new_current_solution

        # Update residual and cost
        if self.current_solution is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(thetas=self.current_solution[0],
                                                                             alphas=self.current_solution[1])
            self.current_sol_cost = torch.norm(self.residual) ** 2
        else:
            self.current_solution, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf

    def fit_several_times(self, n_repetitions=1, forget_current_sol=False):
        """
        Solves the problem `n_repetitions` times, keep and return the best solution.

        Parameters
        ----------
        n_repetitions
            Number of times the algorithm must be repeated.
        forget_current_sol
            Do not consider the current solution of the object. Restart from scratch.

        Returns
        -------
            The best solution.
        """
        # cleaning test this

        # Initialization
        if forget_current_sol:
            # start from scratch
            best_sol, best_sol_cost = None, np.inf
        else:
            # keep current solution as candidate
            best_sol, best_sol_cost = self.current_solution, self.current_sol_cost

        # Main loop, perform independent trials
        for i_repetition in range(n_repetitions):
            self.fit_once()
            self.update_current_solution_and_cost()

            if self.current_sol_cost < best_sol_cost:
                best_sol, best_sol_cost = self.current_solution, self.current_sol_cost

        # Set the current sol to the best one we found
        self.update_current_solution_and_cost(new_current_solution=best_sol)
        return self.current_solution
