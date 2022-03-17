import torch
from torch.nn import functional as f


## 2.2 (diagonal) GMM with Hierarchical Splitting
from pycle.compressive_learning.CLHS import CLHS
from pycle.utils.projectors import ProjectorClip, ProjectorLessUnit2Norm


class CLHS_dGMM(CLHS):

    def __init__(self, sigma2_bar, random_atom, std_lower_bound=1e-10, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Manage bounds
        variance_relative_lower_bound = std_lower_bound ** 2
        variance_relative_upper_bound = 0.5 ** 2

        self.upper_data = torch.ones(self.phi.d, device=self.device, dtype=self.real_dtype)
        self.lower_data = -1. * torch.ones(self.phi.d, device=self.device, dtype=self.real_dtype)
        max_variance = torch.square(self.upper_data - self.lower_data)
        lower_var = variance_relative_lower_bound * max_variance
        upper_var = variance_relative_upper_bound * max_variance

        # Projector
        self.variance_projector = ProjectorClip(lower_var, upper_var)
        self.mean_projector = ProjectorLessUnit2Norm()

        # For initialization of Gaussian atoms
        self.random_atom = random_atom.to(self.device)
        self.sigma2_bar = torch.tensor(sigma2_bar, device=self.device)

    def randomly_initialize_several_atoms(self, nb_atoms): # todo remplacer cette fonction par "initialize one"
        """
        Define how to initialize a number nb_atoms of new atoms.
        :param nb_atoms: int
        :return: torch tensor for new atoms
        """
        # all_new_mu = (self.upper_data - self.lower_data) * torch.rand(nb_atoms, self.freq_matrix.d).to(self.device) + self.lower_data
        all_new_mu = self.random_atom.repeat(nb_atoms, 1)
        all_new_sigma = (1.5 - 0.5) * torch.rand(nb_atoms, self.phi.d, device=self.device) + 0.5
        all_new_sigma *= self.sigma2_bar
        new_theta = torch.cat((all_new_mu, all_new_sigma), dim=1)
        return new_theta

    def sketch_of_atoms(self, thetas):
        """
        Always compute sketch of several atoms.
        Implements Equation 15 of Keriven et al: Sketching for large scale learning of Mixture Models.
        :param thetas: tensor size (n_atoms, d_theta)
        :param freq_matrix: DenseFrequencyMatrix
        :return: tensor size (n_atoms, nb_freq)
        """
        # todo this is the code of a feature map
        # assert isinstance(freq_matrix, DenseFrequencyMatrix)
        # # the 4 following lines do -0.5 * w^T . Sigma . w (right hand part of Eq 15)
        # sigmas_diag = thetas[..., -self.phi.d:]
        # right_hand = sigmas_diag.unsqueeze(-1) * freq_matrix.omega
        # right_hand = freq_matrix.omega * right_hand
        # right_hand = - 0.5 * torch.sum(right_hand, dim=-2)
        # # this line does the multiplication between the frequencies and the means (left hand part of Eq 15)
        # left_hand = -1j * freq_matrix.transpose_apply(thetas[..., :self.phi.d])
        # inside_exp = left_hand + right_hand  # adding the contents of an exp is like multiplying the exps
        # return torch.exp(inside_exp)
        return self.phi(thetas)

    def split_all_current_thetas_alphas(self):
        all_mus, all_sigmas = self.all_thetas[:, :self.phi.d], self.all_thetas[:, -self.phi.d:]
        print(torch.max(all_sigmas, dim=1)[0])
        all_i_max_var = torch.argmax(all_sigmas, dim=1).to(torch.long)
        print(f"Splitting directions: {all_i_max_var}")
        all_direction_max_var = f.one_hot(all_i_max_var, num_classes=self.phi.d)
        all_max_var = all_sigmas.gather(1, all_i_max_var.view(-1, 1)).squeeze()
        all_max_deviation = torch.sqrt(all_max_var)
        all_sigma_step = all_max_deviation.unsqueeze(-1) * all_direction_max_var

        right_splitted_thetas = torch.cat((all_mus + all_sigma_step, all_sigmas), dim=1)
        left_splitted_thetas = torch.cat((all_mus - all_sigma_step, all_sigmas), dim=1)
        self.remove_all_atoms()
        self.add_several_atoms(torch.cat((left_splitted_thetas, right_splitted_thetas), dim=0))

        # Split alphas
        self.alphas = self.alphas.repeat(2) / 2.

    def projection_step(self, theta):
        # Uniform normalization of the variances
        sigma = theta[..., -self.phi.d:]
        self.variance_projector.project(sigma)
        # Normalization of the means
        mu = theta[..., :self.phi.d]
        self.mean_projector.project(mu)

    def get_gmm(self, return_numpy=True):
        """
        Return weights, mus and sigmas as diagonal matrices.
        :param return_numpy: bool
        :return:
        """
        weights = self.alphas
        mus = self.all_thetas[:, :self.phi.d]
        sigmas = self.all_thetas[:, -self.phi.d:]
        sigmas_mat = torch.diag_embed(sigmas)
        if return_numpy:
            return weights.cpu().detach().numpy(), mus.cpu().detach().numpy(), sigmas_mat.cpu().detach().numpy()
        return weights, mus, sigmas_mat
