import torch
import pytest 

from mylib.gvf import create_gvf


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
    return args

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
    x_random = torch.randn((16, 1, 2))
    ts_random = torch.rand(16) * _sigma_max

    for t in ts_random:
        dxdt = gvf(x_random, t)
        assert torch.all(dxdt == torch.zeros((16, 1, 2))), \
                f"For scale = 0 all gradients of gvf should be 0, got : {dxdt}"

#----------------------------------------------------------------------------
# Numerically solved DE should be close to exact solution 
# Apply this to the score of a Gaussian

def _test_score_is_zero_for_scale_zero():
    pass
    # Def solution
    # Match 

