import os

import numpy as np
import torch
from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.pycle_gpu.sketching.frequency_matrix import DenseFrequencyMatrix
from src.pycle_gpu.cl_algo.optimization import ProjectorClip, ProjectorLessUnit2Norm

from pycle.compressive_learning.torch.SolverTorch import SolverTorch


class HierarchicalCLOMP(SolverTorch):
    """
    Stochastic gradient descent during optimization. Simplified solver: we keep only the essential parts of
    the previous code. Hierarchical for the moment.

    Simplification: only l2 projection, and only dense frequency matrix
    """
    def __init__(self, freq_matrix, nb_mixtures, d_theta, sketch, freq_epochs, freq_batch_size, lr, beta_1, beta_2,
                gamma, step_size, verbose):

        super().__init__(phi=)
        assert isinstance(freq_matrix, DenseFrequencyMatrix)
        self.freq_matrix = freq_matrix
        self.nb_mixtures = nb_mixtures
        self.d_theta = d_theta
        self.sketch = sketch
        self.verbose = verbose

        # Optimization parameters
        self.freq_epochs = freq_epochs
        self.freq_batch_size = freq_batch_size
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.step_size = step_size

        self.device = freq_matrix.device
        self.real_dtype = freq_matrix.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128
        self.minimum_atom_norm = 1e-15 * np.sqrt(self.d_theta)

        # Initialization
        if self.verbose:
            print('Initialize empty solution...')
        self.n_atoms = 0
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        if self.verbose:
            print('End of empty solution initialization!')

    # abstract methods
    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError

    def sketch_of_atoms(self, thetas, freq_matrix):
        """
        Always compute sketch of several atoms.
        :param thetas: tensor size (n_atoms, d_theta)
        :param freq_matrix: DenseFrequencyMatrix
        :return: tensor size (n_atoms, nb_freq)
        """
        raise NotImplementedError
        # n_atoms = len(thetas)
        # return torch.empty(n_atoms, self.freq_matrix.m) # + device et tout ?

    def split_all_current_thetas_alphas(self):
        raise NotImplementedError

    def projection_step(self, theta):
        raise NotImplementedError

    def sketch_of_solution(self, alphas, all_thetas, freq_matrix):
        return torch.matmul(alphas.to(self.comp_dtype), self.sketch_of_atoms(all_thetas, freq_matrix))

    def add_several_atoms(self, new_thetas):
        """
        Adding a new atom.
        :param new_thetas: tensor size (n_atoms_to_add, d_atom)
        :return:
        """
        self.n_atoms += len(new_thetas)
        self.all_thetas = torch.cat((self.all_thetas, new_thetas), dim=0)

    def remove_one_atom(self, ind_remove):
        """
        Remove an atom.
        :param ind_remove: list of int
        :return:
        """
        self.n_atoms -= 1
        self.all_thetas = torch.cat((self.all_thetas[:ind_remove], self.all_thetas[ind_remove + 1:]), dim=0)

    def remove_all_atoms(self):
        self.n_atoms = 0
        self.all_thetas = torch.empty(0, self.d_theta, dtype=self.real_dtype).to(self.device)

    def loss_atom_correlation(self, theta, residual):
        sketch_of_atom = torch.squeeze(self.sketch_of_atoms(theta, self.freq_matrix))
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_atom_norm:
            norm_atom = torch.tensor(self.minimum_atom_norm)
        return -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, residual))

    def maximize_atom_correlation(self, residual, tol=1e-2, max_iter=1000, log_dir=None):
        """
        Step 1 in CLOMP-R algorithm. Find most correlated atom. Torch optimization, using Adam.
        :param residual: sketch residual
        :param tol: stopping criteria is to stop when the relative difference of loss between
        two consecutive iterations is less than tol.
        :param max_iter: max iterations number for optimization.
        :param tensorboard: set True to plot loss in Tensorboard.
        :return: updated new_theta
        """
        new_theta = self.randomly_initialize_several_atoms(1).squeeze()
        params = [torch.nn.Parameter(new_theta, requires_grad=True)]
        optimizer = torch.optim.Adam(params, lr=0.01)

        if log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            event_name = current_time + '_' + socket.gethostname()
            comment = "maximize_atom_correlation"
            event_name = event_name + "_" + comment
            writer = SummaryWriter(os.path.join(log_dir, event_name))

        for i in range(max_iter):
            optimizer.zero_grad()
            loss = self.loss_atom_correlation(params[0], residual)
            loss.backward()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                self.projection_step(new_theta)
            if i == 0:
                previous_loss = torch.clone(loss)
            else:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if log_dir:
                    writer.add_scalar(f'maximize_atom_correlation', loss.item(), i)
                if relative_loss_diff.item() <= tol:
                    break
                previous_loss = torch.clone(loss)
        if log_dir:
            writer.flush()
            writer.close()
        return new_theta.data.detach()

    def minimize_cost_from_current_sol(self, log_dir=None, sub_log_dir=None):
        # Preparing frequencies dataloader
        dataset = TensorDataset(torch.transpose(self.freq_matrix.omega, 0, 1), self.sketch)
        dataloader = DataLoader(dataset, batch_size=self.freq_batch_size)

        # Parameters, optimizer
        log_alphas = torch.log(self.alphas).requires_grad_()
        all_thetas = self.all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]
        # Adam optimizer
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(self.beta_1, self.beta_2))

        if self.step_size > 0:
            scheduler = StepLR(optimizer, self.step_size, gamma=self.gamma)

        # Tensorboard
        if log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            event_name = current_time + '_' + socket.gethostname()
            if sub_log_dir:
                event_name = event_name + "_" + sub_log_dir
            comment = f'BATCH_{self.freq_batch_size}_#EPOCHS_{self.freq_epochs}_LR_{self.lr}_BETA1_{self.beta_1}_BETA2_{self.beta_2}_GAMMA_{self.gamma}_STEP_SIZE_{self.step_size}'
            event_name = event_name + "_" + comment
            writer = SummaryWriter(os.path.join(log_dir, event_name))

        for i in range(self.freq_epochs):
            for iter_batch, (freq_transpose_batch, sketch_batch) in enumerate(dataloader):
                # freq_batch is of size (freq_batch_size, nb_freq)
                reduced_freq_mat = DenseFrequencyMatrix(self.freq_matrix.d, self.freq_matrix.m, self.freq_matrix.device, self.freq_matrix.dtype)
                reduced_freq_mat.omega = torch.transpose(freq_transpose_batch, 0, 1)
                reduced_sketch_of_sol = self.sketch_of_solution(torch.exp(log_alphas), all_thetas, reduced_freq_mat)
                loss = torch.square(torch.linalg.norm(sketch_batch - reduced_sketch_of_sol))
                loss.backward()
                optimizer.step()
                # Projection step
                with torch.no_grad():
                    self.projection_step(all_thetas)
                # Tracking loss
                if log_dir:
                    writer.add_scalar(f"minimize_cost_from_current_sol/epoch-{i}", loss.item(), iter_batch)
                # Scheduler step
                if self.step_size > 0:
                    scheduler.step()
        if log_dir:
            writer.flush()
            writer.close()
        self.alphas = torch.exp(log_alphas).detach()
        self.all_thetas = all_thetas.detach()

    def fit_once(self, runs_dir=None):
        n_iterations = int(np.ceil(np.log2(self.nb_mixtures)))  # log_2(K) iterations
        # new_theta = self.maximize_atom_correlation(self.sketch, log_dir=runs_dir)
        new_theta = self.randomly_initialize_several_atoms(1).squeeze()
        self.add_several_atoms(torch.unsqueeze(new_theta, 0))
        self.alphas = torch.ones(1, dtype=self.real_dtype).to(self.device)

        for i_iter in range(n_iterations):
            if self.verbose:
                print(f'Iteration {i_iter + 1} / {n_iterations}')
            print("Splitting all atoms...")
            self.split_all_current_thetas_alphas()
            print("Fine-tuning...")
            self.minimize_cost_from_current_sol(log_dir=runs_dir, sub_log_dir=f'ITER_{i_iter + 1}')

        print("Final fine-tuning...")
        self.minimize_cost_from_current_sol(log_dir=runs_dir, sub_log_dir='FINAL_FINE_TUNING')
        self.projection_step(self.all_thetas)
        self.alphas /= torch.sum(self.alphas)