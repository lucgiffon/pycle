import time
from abc import abstractmethod
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f

from pycle.compressive_learning.torch.SolverTorch import SolverTorch
from pycle.utils.optim import ObjectiveValuesStorage


class CLOMP(SolverTorch):
    """
    Template for a compressive learning solver, using torch implementation, to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.

    Some size of tensors to keep in mind:
        - alphas: (n_atoms,)-tensor, weigths of the mixture elements
        - all_thetas:  (n_atoms,d_atom)-tensor, all the found parameters in matrix form
        - all_atoms: (m,n_atoms)-tensor, the sketch of the found parameters (m is sketch size)
    """

    def __init__(self, phi, nb_mixtures, d_atom, bounds, sketch, sketch_weight=1., verbose=False,
                 lr_inner_optimizations=0.01, maxiter_inner_optimizations=1000, tol_inner_optimizations=1e-5,
                 show_curves=False, tensorboard=False, path_template_tensorboard_writer="CLOMP/{}/loss/"):
        """
        - phi: a FeatureMap object
        - sketch: tensor
        - sketch_weight: float, a re-scaling factor for the data sketch
        - verbose: bool
        """
        super().__init__(phi=phi, nb_mixtures=nb_mixtures, d_atom=d_atom, bounds=bounds, sketch=sketch, sketch_weight=sketch_weight, verbose=verbose)

        # Other minor params
        self.lr_inner_optimizations = lr_inner_optimizations
        self.maxiter_inner_optimizations = maxiter_inner_optimizations
        self.tol_inner_optimizations = tol_inner_optimizations
        self.path_template_tensorboard_writer = path_template_tensorboard_writer
        self.show_curves = show_curves
        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter()

    # Abstract methods
    # ===============
    # Methods that have to be instantiated by child classes

    @abstractmethod
    def sketch_of_atoms(self, all_theta):
        """
        Computes and returns A_Phi(P_theta_k) for all theta_k. Do the computation with torch.
        Return size (m) or (K, m)
        """
        raise NotImplementedError

    @abstractmethod
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError

    @abstractmethod
    def randomly_initialize_new_atom(self):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError

    @abstractmethod
    def projection_step(self, theta):
        raise NotImplementedError

    # Generic methods
    # ===============
    # Methods that are general for all instances of this class
    # Instantiation of methods of parent class

    def initialize_empty_solution(self):
        self.n_atoms = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)
        self.all_atoms = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)

        self.current_sol = (self.alphas, self.all_thetas)  # Overwrite

    def sketch_of_solution(self, solution=None):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).
        In: solution = (all_thetas, alphas)
            phi = sk.ComplexExpFeatureMap
            one_by_one = compute one atom by one atom in case atom computation does not fit in GPU
        Out: sketch_of_solution: (m,)-tensor containing the sketch
        """
        if solution is None:
            all_thetas, alphas = self.all_thetas, self.alphas
        else:
            all_thetas, alphas = solution
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

    def remove_atom(self, index_to_remove):
        """
        Remove an atom.
        :param index_to_remove: int
        :return:
        """
        self.n_atoms -= 1
        self.all_thetas = torch.cat((self.all_thetas[:index_to_remove], self.all_thetas[index_to_remove+1:]), dim=0)
        self.all_atoms = torch.cat((self.all_atoms[:, :index_to_remove], self.all_atoms[:, index_to_remove + 1:]),
                                   dim=1)

    def replace_atom(self, index_to_replace, new_theta):
        """
        Replace an atom
        :param index_to_replace: int
        :param new_theta: tensor
        :return:
        """
        self.all_thetas[index_to_replace] = new_theta
        self.all_atoms[:, index_to_replace] = self.sketch_of_atoms(new_theta)

    def loss_atom_correlation(self, theta):
        sketch_of_atom = self.sketch_of_atoms(theta)
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_atom_norm:
            norm_atom = torch.tensor(self.minimum_atom_norm)
        return -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, self.residual))

    def maximize_atom_correlation(self, new_theta):
        """
        Step 1 in CLOMP-R algorithm. Find most correlated atom. Torch optimization, using Adam.

        :param new_theta: torch tensor for atom
        :param tol: stopping criteria is to stop when the relative difference of loss between
        two consecutive iterations is less than tol.
        :param max_iter: max iterations number for optimization.
        :return: updated new_theta
        """
        params = [torch.nn.Parameter(new_theta, requires_grad=True)]
        optimizer = torch.optim.Adam(params, lr=self.lr_inner_optimizations)


        for i in range(self.maxiter_inner_optimizations):
            optimizer.zero_grad()

            loss = self.loss_atom_correlation(params[0])

            loss.backward()
            optimizer.step()

            # Projection step
            with torch.no_grad():
                self.projection_step(new_theta)

            if i != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if self.show_curves:
                    ObjectiveValuesStorage().add(float(previous_loss), "maximize_atom_correlation")
                if self.tensorboard:
                    self.writer.add_scalar(self.path_template_tensorboard_writer.format("step1"), loss.item(), i)
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

    def find_optimal_weights(self, normalize_atoms=False):
        """
        Using the current atoms matrix, find the optimal weights for the sketch of the mixture to
        approximate well the true sketch. (Step 4)

        2nd optimization sub-problem in CLOMP
        """
        log_alphas = torch.nn.Parameter(torch.zeros(self.n_atoms, device=self.device), requires_grad=True)
        optimizer = torch.optim.Adam([log_alphas], lr=self.lr_inner_optimizations)

        if normalize_atoms:
            all_atoms = f.normalize(self.all_atoms, dim=1, eps=self.minimum_atom_norm)
        else:
            all_atoms = self.all_atoms

        for i in range(self.maxiter_inner_optimizations):
            optimizer.zero_grad()

            sketch_solution = torch.matmul(all_atoms, torch.exp(log_alphas).to(self.comp_dtype))
            loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution)

            loss.backward()
            optimizer.step()

            if i != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if self.tensorboard:
                    self.writer.add_scalar(self.path_template_tensorboard_writer.format('step3-4'), loss.item(), i)
                if self.show_curves:
                    ObjectiveValuesStorage().add(float(previous_loss), "find_optimal_weights")

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

    def minimize_cost_from_current_sol(self):
        """
        Step 5 in CLOMP-R algorithm. At the end of the method, update the parameters self.alphas and self.all_thetas.

        Third optimization subproblem.

        :param tol: float. Stopping criteria for optimization: when the relative difference in loss is less than tol.
        :param max_iter: int. Maximum number of iterations.
        :param tensorboard: set True to plot loss in Tensorboard.
        :return:
        """
        # Parameters, optimizer
        log_alphas = torch.log(self.alphas).requires_grad_()
        all_thetas = self.all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]
        optimizer = torch.optim.Adam(params, lr=self.lr_inner_optimizations)

        for iteration in range(self.maxiter_inner_optimizations):
            optimizer.zero_grad()
            # Designing loss
            alphas = torch.exp(log_alphas)

            sketch_solution = self.sketch_of_solution((all_thetas, alphas))
            loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution)

            loss.backward()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                self.projection_step(all_thetas)

            # Tracking loss
            if iteration != 0:
                relative_loss_diff = torch.abs(previous_loss - loss) / previous_loss
                if self.tensorboard:
                    self.writer.add_scalar(self.path_template_tensorboard_writer.format('step5'), loss.item(), iteration)
                if self.show_curves:
                    ObjectiveValuesStorage().add(float(previous_loss), "minimize_cost_from_current_sol")

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
        self.residual = self.sketch_reweighted - self.sketch_of_solution((self.all_thetas, self.alphas))

    def final_fine_tuning(self):
        logger.info(f'Final fine-tuning...')
        self.minimize_cost_from_current_sol()
        # self.projection_step(self.all_thetas) # this is useless given the projection was made in the last method
        logger.debug(torch.norm(self.all_thetas[:, :self.phi.d], dim=1))
        self.alphas /= torch.sum(self.alphas)
        self.current_sol = (self.alphas, self.all_thetas)

    def fit_once(self):
        """
        CLOMP-R algorithm implementation.
        If random_restart is True, constructs a new solution from scratch with CLOMP-R, else fine-tune.
        """
        n_iterations = 2 * self.nb_mixtures

        for i_iter in range(n_iterations):
            logger.debug(f'Iteration {i_iter + 1} / {n_iterations}')
            # Step 1: find new atom theta most correlated with residual
            new_theta = self.randomly_initialize_new_atom()

            since = time.time()
            new_theta = self.maximize_atom_correlation(new_theta)
            logger.debug(f'Time for step 1: {time.time() - since}')

            # Step 2: add it to the support
            self.add_atom(new_theta)

            # Step 3: if necessary, hard-threshold to enforce sparsity
            if self.n_atoms > self.nb_mixtures:
                since = time.time()
                beta = self.find_optimal_weights(normalize_atoms=True)
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_atom(index_to_remove)
                logger.debug(f'Time for step 3: {time.time() - since}')
                if index_to_remove == self.nb_mixtures:
                    continue

            # Step 4 and 5
            self.do_step_4_5()

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
