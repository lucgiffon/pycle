from abc import ABCMeta, abstractmethod

import numpy as np

from pycle.sketching.feature_maps.FeatureMap import FeatureMap


class Solver(metaclass=ABCMeta):
    """
    Template for a compressive learning solver, used to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.
    Implements several trials of an generic method and keeps the best one.
    """

    def __init__(self, phi, nb_mixtures, d_theta, bounds, sketch=None, sketch_weight=1., verbose=0):
        """
        - Phi: a FeatureMap object
        - sketch_weight: float, a re-scaling factor for the data sketch
        """

        # Encode feature map
        assert isinstance(phi, FeatureMap)
        self.phi = phi

        # Set other values
        self.nb_mixtures = nb_mixtures
        self.d_theta = d_theta
        self.n_atoms = 0

        # Encode sketch and sketch weight
        self.sketch = sketch
        self.update_sketch_and_weight(sketch, sketch_weight)

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


    # Abtract methods
    # ===============
    # Methods that have to be instantiated by child classes
    @abstractmethod
    def initialize_empty_solution(self) -> None:
        """
        This method prepares the attributes that will contain the solution of the compressive learning problem.
        :return:
        """
        pass

    @abstractmethod
    def set_bounds_atom(self, bounds):
        pass
    
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
    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""
        raise NotImplementedError


    @abstractmethod
    def sketch_of_atoms(self, thetas):
        """
        Always compute sketch of several atoms.
        :param thetas: tensor size (n_atoms, d_theta)
        :return: tensor size (n_atoms, nb_freq)
        """
        raise NotImplementedError

    # Generic methods
    # ===============
    # They should always work, using the instances of the methdos above
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