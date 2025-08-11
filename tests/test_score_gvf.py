import torch
import pytest 

from mylib.gvf import create_gvf
from dnnlib.util import to_easydict

@pytest.fixture
def ambient_args():
    # Inialize batch of 2D gvf's
    means = torch.randn((16, 1, 2))      #(batch, num_examples, dim)
    args = {
        "latent" : "ambient", 
        "scale" : 1.0 , 
        "vectorfield" : { 
            "features_template"  : means,
            "args_noise" : "edm", 
            "noise_gate" : {"type_gate" : "quadratic", "nu" : 30.0, },
        },
        "noise" : "edm",
    }
    return to_easydict(args)

#----------------------------------------------------------------------------
# When passing (B, N, D1, D2, ....) : N == 1 use score for single feature
# When passing (B, N, D1, D2, ....) : N > 1 use score with attention

def test_number_of_templates_match_type_of_score(ambient_args):

    args_single = ambient_args.copy()
    args_single["vectorfield"] = ambient_args["vectorfield"].copy()
    args_single["vectorfield"]["features_template"] = torch.rand(16, 1, 3, 64, 64)

    args_multiple = ambient_args.copy()
    args_multiple["vectorfield"] = ambient_args["vectorfield"].copy()
    args_multiple["vectorfield"]["features_template"] = torch.rand(16, 5, 3, 64, 64)

    gvf_single = create_gvf(**args_single)
    gvf_multiple = create_gvf(**args_multiple)

    assert gvf_single.vf_latent._score.__name__ == "_score_single_feature"
    assert gvf_multiple.vf_latent._score.__name__ == "_score_attention"


#----------------------------------------------------------------------------
# For scale = 0, the reverse SDE should equal 0 
# We test on on a batch of 2D examples
# TODO: test this for nonlinear

def test_score_is_zero_for_scale_zero(ambient_args):
    _sigma_max = 80 

    args = ambient_args.copy()
    args["scale"] = 0.0

    gvf = create_gvf(**args, device="cpu")

    # Sample random points and times
    x_random = torch.randn((16, 2))
    ts_random = torch.rand(16) * _sigma_max

    for t in ts_random:
        dxdt = gvf(x_random, t)
        assert torch.all(dxdt == torch.zeros((16, 2))), \
                f"For scale = 0 all gradients of gvf should be 0, got : {dxdt}"

#----------------------------------------------------------------------------
# Numerically solved DE should be close to exact solution 
# Apply this to the score of a Gaussian

class SolverGaussianExact:
    """Solver for a Revers-SDE Gaussian in the EDM probablity flow ODE formulation"""
    def __init__(self, x0, T, nu):
        self.x0 = x0
        self.T = T
        self.nu = nu

    def __call__(self, x_T, t):
        coefficient = torch.sqrt(t**2 + self.nu**2) / torch.sqrt(self.T**2 / self.nu**2)
        return (1 - coefficient) * self.x0 + coefficient * x_T


def _test_solved_reversed_sde_matches_exact_solution(ambient_args):
    args = ambient_args

    # Determinine parameters of Gaussian and inital conditions
    batch_size = 3
    nu = torch.rand(1)
    examples = torch.randn(batch_size, 1, 2)
    T = 80

    # Initialize exact solver and reverse-sde
    solver_exact = SolverGaussianExact(x0=examples, nu=nu, T=T)
    args.vectorfield.features_template = examples
    args.vectorfield.noise_gate.nu = nu
    gvf = create_gvf(**args)

    # Solve the reverse-SDE:
    x_Ts = torch.rand(batch_size, 1, 2) * T
    t_steps = torch.linspace(T, 0, steps=100)
    x_next = x_Ts
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        dt = t_next - t_cur 
        x_next += gvf(x_next, t_cur) * dt

        assert torch.all(x_next == solver_exact(x_Ts, t_next))


def test_solved_reversed_sde_convereges_to_example_nu_equal_zero(ambient_args):
    args = ambient_args

    # Setup examples and initial XT
    T = 80
    batch_size = 2
    x_T = torch.rand(batch_size, 2) * T         # (batch, data)
    examples = torch.rand(batch_size, 1, 2)     # (batch, examples, data)

    # Initialize exact solver and reverse-sde
    args.vectorfield.features_template = examples
    args.vectorfield.noise_gate.nu = 0
    gvf = create_gvf(**args)

    # Solve the reverse-SDE:
    t_steps = torch.linspace(T, 0, steps=100)
    xs = torch.empty(len(t_steps), batch_size, 2)
    xs[0] = x_T
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        dt = t_next - t_cur 
        xs[i+1] = xs[i] + gvf(xs[i], t_cur) * dt

    assert torch.all(torch.isclose(xs[-1], args.vectorfield.features_template.squeeze(1)))

