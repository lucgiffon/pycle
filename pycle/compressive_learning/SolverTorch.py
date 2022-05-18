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
    Template for a compressive learning solver, used to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.
    Implements several trials of an generic method and keeps the best one.
    """
    def __init__(self, phi: FeatureMap, sketch, nb_mixtures, d_theta, bounds,
                 show_curves: bool = False, tensorboard: bool = False,
                 path_template_tensorboard_writer="CLOMP/{}/loss/", dct_opt_method: [None, dict] = None,
                 sketch_weight=1., verbose=0,
                 *args, **kwargs):
        """
        - Phi: a FeatureMap object
        - sketch_weight: float, a re-scaling factor for the data sketch
        """
        # Attributes related to the optimization method used
        self.dct_opt_method = dct_opt_method or dict()  # todo utiliser le dicitonnaire d'optim
        self.initialize_parameters_optimization()

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch.device.type
        self.device = phi.device
        self.real_dtype = phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        # parameters regarding the tracking of the algorithm
        self.show_curves = show_curves
        self.tensorboard = tensorboard
        self.path_template_tensorboard_writer = path_template_tensorboard_writer
        if tensorboard:
            self.writer = SummaryWriter()

        # Encode feature map
        assert isinstance(phi, FeatureMap)
        self.phi = phi

        # Set other values
        self.nb_mixtures = nb_mixtures
        self.d_theta = d_theta
        self.n_atoms = 0

        # Encode sketch and sketch weight
        self.sketch = sketch
        self.sketch_weight = None
        self.sketch_reweighted = None
        self.update_sketch_and_weight(sketch, sketch_weight)

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

        # Verbose
        self.verbose = verbose

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

    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution.
        If `sol` given, also updates the `current_sol` attribute."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(all_thetas=self.all_thetas,
                                                                             alphas=self.alphas)
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
        pass

    @abstractmethod
    def sketch_of_atoms(self, thetas):
        """
        Always compute sketch of several atoms.
        :param thetas: tensor size (n_atoms, d_theta)
        :return: tensor size (n_atoms, nb_freq)
        """
        raise NotImplementedError

    @abstractmethod
    def sketch_of_solution(self, alphas, all_thetas=None, all_atoms=None):
        """
        Should return the sketch of the given solution, A_Phi(P_theta).

        In: a solution P_theta (the exact encoding is determined by child classes), if None use the current sol
        Out: sketch_of_solution: (m,)-array containing the sketch
        """
        raise NotImplementedError

    @abstractmethod
    def fit_once(self, random_restart=False):
        """Optimizes the cost to the given sketch, by starting at the current solution"""
        raise NotImplementedError

    @abstractmethod
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError

    # Generic methods
    # ===============
    def update_sketch_and_weight(self, sketch=None, sketch_weight=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""
        if sketch is not None:
            self.sketch = sketch
        if sketch_weight is not None:
            assert isinstance(sketch_weight, float) or isinstance(sketch_weight, int)
            self.sketch_weight = sketch_weight
        self.sketch_reweighted = self.sketch_weight * self.sketch

    def fit_several_times(self, n_repetitions=1, forget_current_sol=False):
        """Solves the problem n times. If a sketch is given, updates it."""

        # Initialization
        if forget_current_sol:
            # start from scratch
            best_sol, best_sol_cost = None, np.inf
        else:
            # keep current solution as candidate
            best_sol, best_sol_cost = self.current_sol, self.current_sol_cost

        # Main loop, perform independent trials
        for i_repetition in range(n_repetitions):
            self.fit_once(random_restart=True)
            self.update_current_sol_and_cost()

            if self.current_sol_cost < best_sol_cost:
                best_sol, best_sol_cost = self.current_sol, self.current_sol_cost

        # Set the current sol to the best one we found
        self.current_sol, self.current_sol_cost = best_sol, best_sol_cost
        return self.current_sol
