import numpy as np

from pycle.sketching import FeatureMap


# 0.1 Generic solver (stores a sketch and a solution, can run multiple trials of a learning method to specify)
class Solver:
    """
    Template for a compressive learning solver, used to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.
    Implements several trials of an generic method and keeps the best one.
    """

    def __init__(self, Phi, sketch=None, sketch_weight=1., verbose=0):
        """
        - Phi: a FeatureMap object
        - sketch_weight: float, a re-scaling factor for the data sketch
        """

        # Encode feature map
        assert isinstance(Phi, FeatureMap)
        self.Phi = Phi

        # Encode sketch and sketch weight
        self.update_sketch_and_weight(sketch, sketch_weight)

        # Encode current theta and cost value
        self.update_current_sol_and_cost(None)

        # Verbose
        self.verbose = verbose

    # Abtract methods
    # ===============
    # Methods that have to be instantiated by child classes
    def sketch_of_solution(self, sol=None):
        """
        Should return the sketch of the given solution, A_Phi(P_theta).

        In: a solution P_theta (the exact encoding is determined by child classes), if None use the current sol
        Out: sketch_of_solution: (m,)-array containing the sketch
        """
        raise NotImplementedError

    def fit_once(self, random_restart=False):
        """Optimizes the cost to the given sketch, by starting at the current solution"""
        raise NotImplementedError

    # Generic methods
    # ===============
    # They should always work, using the instances of the methdos above
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

    def update_sketch_and_weight(self, sketch=None, sketch_weight=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""
        if sketch is not None:
            self.sketch = sketch
        if sketch_weight is not None:
            assert isinstance(sketch_weight, float) or isinstance(sketch_weight, int)
            self.sketch_weight = sketch_weight
        self.sketch_reweighted = self.sketch_weight * self.sketch

    def update_current_sol_and_cost(self, sol=None):
        """Updates the residual and cost to the current solution. If sol given, also updates it."""

        # Update current sol if argument given
        if sol is not None:
            self.current_sol = sol

        # Update residual and cost
        if self.current_sol is not None:
            self.residual = self.sketch_reweighted - self.sketch_of_solution(self.current_sol)
            self.current_sol_cost = np.linalg.norm(self.residual)
        else:
            self.current_sol, self.residual = None, self.sketch_reweighted
            self.current_sol_cost = np.inf