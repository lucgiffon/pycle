from abc import abstractmethod

import torch
import numpy as np
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from pycle.compressive_learning.SolverTorch import SolverTorch
from pycle.sketching.feature_maps.GMMFeatureMap import GMMFeatureMap


class CLHS(SolverTorch):
    """
    Stochastic gradient descent during optimization. Simplified solver: we keep only the essential parts of
    the previous code. Hierarchical for the moment.

    Simplification: only l2 projection, and only dense frequency matrix
    """
    def __init__(self, phi, freq_batch_size,
                gamma=0.98, step_size=1, *args, **kwargs):

        super().__init__(phi=phi, d_theta=phi.d*2, *args, **kwargs)

        # Optimization parameters
        self.freq_batch_size = freq_batch_size

        self.gamma = gamma
        self.step_size = step_size

    @abstractmethod
    def split_all_current_thetas_alphas(self):
        raise NotImplementedError

    def projection_step(self, theta):
        raise NotImplementedError

    def set_bounds_atom(self, bounds):
        self.bounds = None  # for compatibility

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

    def minimize_cost_from_current_sol(self, prefix=""):
        # Preparing frequencies dataloader
        dataset = TensorDataset(torch.transpose(self.phi.Omega, 0, 1), self.sketch)
        dataloader = DataLoader(dataset, batch_size=self.freq_batch_size)

        # Parameters, optimizer
        log_alphas = torch.log(self.alphas).requires_grad_()
        all_thetas = self.all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]
        # Adam optimizer
        optimizer = torch.optim.Adam(params, lr=self.lr_inner_optimizations, betas=(self.beta_1, self.beta_2))

        if self.step_size > 0:
            scheduler = StepLR(optimizer, self.step_size, gamma=self.gamma)
        print("SALUT")
        for i in range(self.maxiter_inner_optimizations):
            for iter_batch, (freq_transpose_batch, sketch_batch) in enumerate(dataloader):
                # freq_batch is of size (freq_batch_size, nb_freq)
                reduced_phi = GMMFeatureMap(self.phi.name, torch.transpose(freq_transpose_batch, 0, 1), device=self.phi.device, use_torch=True)

                reduced_sketch_of_sol = torch.matmul(torch.exp(log_alphas).to(self.comp_dtype), reduced_phi(all_thetas))
                loss = torch.square(torch.linalg.norm(sketch_batch - reduced_sketch_of_sol))

                loss.backward()
                optimizer.step()
                # Projection step
                with torch.no_grad():
                    self.projection_step(all_thetas)
                # Tracking loss
                if self.tensorboard:
                    comment = f'{prefix}/BATCH_{self.freq_batch_size}_#EPOCHS_{self.maxiter_inner_optimizations}_LR_{self.lr_inner_optimizations}_BETA1_{self.beta_1}_BETA2_{self.beta_2}_GAMMA_{self.gamma}_STEP_SIZE_{self.step_size}'
                    self.writer.add_scalar(f"minimize_cost_from_current_sol/{comment}/epoch-{i}", loss.item(), iter_batch)
                # Scheduler step
                if self.step_size > 0:
                    scheduler.step()

        if self.tensorboard:
            self.writer.flush()
            self.writer.close()

        self.alphas = torch.exp(log_alphas).detach()
        self.all_thetas = all_thetas.detach()

    def fit_once(self, runs_dir=None):
        n_iterations = int(np.ceil(np.log2(self.nb_mixtures)))  # log_2(K) iterations
        # new_theta = self.maximize_atom_correlation(self.sketch, log_dir=runs_dir)
        new_theta = self.randomly_initialize_several_atoms(1).squeeze()
        self.add_several_atoms(torch.unsqueeze(new_theta, 0))
        self.alphas = torch.ones(1, dtype=self.real_dtype).to(self.device)

        for i_iter in range(n_iterations):
            logger.debug(f'Iteration {i_iter + 1} / {n_iterations}')
            logger.debug("Splitting all atoms...")
            self.split_all_current_thetas_alphas()
            logger.debug("Fine-tuning...")
            self.minimize_cost_from_current_sol(prefix=str(i_iter))
            self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

        logger.debug("Final fine-tuning...")
        self.minimize_cost_from_current_sol(prefix='FINAL_FINE_TUNING')
        self.projection_step(self.all_thetas)
        self.alphas /= torch.sum(self.alphas)
        self.update_current_sol_and_cost(sol=(self.all_thetas, self.alphas))

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