from pycle.utils.torch_functions import MultiSigmaARFrequencyMatrixLinApEncDec, LinearFunctionEncDec
import torch
import pytest

from pycle.utils.intermediate_storage import ObjectiveValuesStorage


@pytest.fixture
def dim():
    return 10


def function_optim(loss, p_dim):
    for j in range(7):
        x = torch.nn.Parameter(torch.randn((1, p_dim), requires_grad=True))

        optim = torch.optim.Adam([x], lr=0.1)
        for i in range(100):
            optim.zero_grad()
            loss_res = loss(x)
            ObjectiveValuesStorage().add(float(loss_res), f"loss_{j}")
            loss_res.backward()
            optim.step()

        print(loss(x))
    ObjectiveValuesStorage().show(title="Torch simple optimisation with Adam")
    ObjectiveValuesStorage().clear()


def test_LinearFunctionEncDec(dim):
    target = torch.unsqueeze(torch.randn(dim, requires_grad=False), 0)
    w = torch.eye(dim, requires_grad=False)
    def loss(x): return (LinearFunctionEncDec.apply(x, w, False, False) - target).pow(2).sum()
    function_optim(loss, dim)


def test_MultiSigmaARFrequencyMatrixLinApEncDec(dim):
    nb_sigmas = 2
    nb_replicates = 5
    target = torch.unsqueeze(torch.randn(dim * nb_sigmas * nb_replicates, requires_grad=False), 0)

    # directions = torch.eye(dim, requires_grad=False)
    # SigFacts = torch.tensor([1.] * nb_sigmas)
    # R = torch.ones((dim, nb_replicates))
    directions = torch.randn((dim, dim), requires_grad=False)
    SigFacts = torch.randn((nb_sigmas,))
    R = torch.randn((dim, nb_replicates))

    def loss(x):
        return (MultiSigmaARFrequencyMatrixLinApEncDec.apply(x, SigFacts, directions, R, False, False) - target).pow(2).sum()

    function_optim(loss, dim)
