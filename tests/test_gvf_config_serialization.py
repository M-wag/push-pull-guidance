import itertools
import random
import pytest
import torch

from mylib.helpers import load_from_json, save_to_json
from mylib.gvf import args_is_linear, create_gvf
from training.networks import EDMPrecond

#----------------------------------------------------------------------------
# Dummy version for testing EDMPrecond

@pytest.fixture()
def edm_net():
    return EDMPrecond(img_resolution=16, img_channels=3)

#----------------------------------------------------------------------------
# Options which generate possible GuidanceVectorField configurations

LATENT_ARGS = [
         "ambient",
         {"seed" : random.randint(0,100), "dim_in" : random.randint(16,32),
          "dim_out" : random.randint(8, 16), "n_features" : random.randint(1,3)},
         {"net": "__REF__network", "attribute" : "attention", "index" : [-2, -1]},
         {"net": "__REF__network", "attribute" : "skip", "index" : [-2, -1]},
         {"autoencoder" : "tiny", "id" : "madebyollin/taesd"}
    ]

NOISE_GATE_ARGS = [
        {"type_gate" : "quadratic", "nu" : random.randint(0, 80), "noise_onset" : random.random() * 80},
        {"type_gate" : "logistic", "nu" : random.randint(0, 80), "decay_rate" :  random.random(), "noise_onset" : random.random() * 80},
    ]

ARGS_NOISE_ARGS = ["edm"]

VECTORFIELD_ARGS = [
    {
        "noise_gate": gate, 
        "args_noise": noise, 
        "features_template" : "__REF__features_template"
     }
    for gate, noise in itertools.product(
        NOISE_GATE_ARGS, ARGS_NOISE_ARGS
        )
    ]

PULLBACK_ARGS = [
        None,
        "jvp",
        {"step_size_intercept" : 1, "step_size_slope" : 0}
    ]


ALL_CONFIGS = list(itertools.product(
        LATENT_ARGS,
        VECTORFIELD_ARGS,
        PULLBACK_ARGS,
        ARGS_NOISE_ARGS,
    ))


#----------------------------------------------------------------------------
# Assert config == load(save(config))

def _test_json_save_and_load_equals_identity(config):
    assert config == load_from_json(save_to_json(config))

#----------------------------------------------------------------------------
# Assert config_out = create_config(create_gvf(config_in))

@pytest.mark.parametrize('args_latent, args_vectorfield, args_pullback, args_noise', ALL_CONFIGS)
def test_config_gvf_in_equals_config_gvf_out(
        args_latent,
        args_vectorfield,
        args_pullback,
        args_noise,
        edm_net,
    ):

    # Add non-seriazible values which need to be referenced
    args_references = {}
    args_references["features_template"] = torch.zeros((1, 3, 2, 2))
    args_references["network"] = edm_net

    # Construct gvf args
    args_in = {
        "latent"        : args_latent,
        "vectorfield"   : args_vectorfield,
        "pullback"      : args_pullback,
        "noise"         : args_noise,
        "scale"         : round(random.random(), 3)
    }

    # When a linear latent and a pulback are specified throw a error
    # Linear latents always use the "linear" pullback
    should_raise = args_is_linear(args_latent) and args_pullback is not None
    if should_raise:
        with pytest.raises(ValueError):
            create_gvf(**args_in, args_references=args_references)
        return

    gvf = create_gvf(**args_in, args_references=args_references)

    # Check whether two configs are identitcal
    args_out = gvf.args

    assert args_in == args_out, f"Expected : \n {args_in} \n\n Outcome: {args_out}\n"

#----------------------------------------------------------------------------
# Test for nested GuidanceVectorfield



