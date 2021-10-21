import numpy as np

from pycle.compressive_learning.numpy import CLOMP_dGMM


## 2.2 (diagonal) GMM with Hierarchical Splitting
class CLHS_dGMM(CLOMP_dGMM):
    """
    CL Hierarchical Splitting solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K Gaussians
    with diagonal covariances to the sketch.
    Due to strong overlap, this algorithm is strongly based on CLOMP for GMM algorithm (its the parent class),
    but the core fitting method is overridden.
    Requires the feature map to be Fourier features.
    """

    def __init__(self, Phi, K, bounds, sketch=None, sketch_weight=1., init_variance_mode="sketch", verbose=0):
        super(CLHS_dGMM, self).__init__(Phi, K, bounds, sketch, sketch_weight, init_variance_mode, verbose)

    # New split methods
    def split_one_atom(self, k):
        """Splits the atom at index k in two.
        The first result of the split is replaced at the k-th index,
        the second result is added at the end of the atom list."""

        # Pick the dimension with most variance
        theta_k = self.Theta[k]
        (mu, sig) = (theta_k[:self.Phi.d], theta_k[-self.Phi.d:])
        i_max_var = np.argmax(sig)

        # Direction and stepsize
        direction_max_var = np.zeros(self.Phi.d)
        direction_max_var[i_max_var] = 1.  # i_max_var-th canonical basis vector in R^d
        SD_max = np.sqrt(sig[i_max_var])  # max standard deviation

        # Split!
        self.add_atom(np.append(mu + SD_max * direction_max_var, sig))  # "Right" split
        self.replace_atom(k, np.append(mu - SD_max * direction_max_var, sig))  # "Left" split

    def split_all_atoms(self):
        """Self-explanatory"""
        for k in range(self.n_atoms):
            self.split_one_atom(k)

    # Override the main fit_once method
    def fit_once(self, random_restart=True):
        """
        If random_restart is True, constructs a new solution from scratch with CLHS, else fine-tune.
        """

        if random_restart:
            ## Main mode of operation

            # Initializations
            n_iterations = int(np.ceil(np.log2(self.K)))  # log_2(K) iterations
            self.initialize_empty_solution()
            self.residual = self.sketch_reweighted

            # Add the starting atom
            new_theta = self.randomly_initialize_new_atom()
            new_theta = self.maximize_atom_correlation(new_theta)
            self.add_atom(new_theta)

            # Main loop
            for i_iteration in range(n_iterations):
                ## Step 1-2: split the currently selected atoms
                self.split_all_atoms()

                ## Step 3: if necessary, hard-threshold to enforce sparsity
                while self.n_atoms > self.K:
                    beta = self.find_optimal_weights(normalize_atoms=True)
                    index_to_remove = np.argmin(beta)
                    self.remove_atom(index_to_remove)

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