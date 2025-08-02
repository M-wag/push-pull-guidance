import pytest
import random
import itertools
import torch

from mylib.helpers import load_from_json, save_to_json
from mylib.diffusion import load_templates_batch
from mylib.gvf_2 import args_is_linear, match_args_to_pullback, create_gvf

#----------------------------------------------------------------------------
# Helper function to compare two nested dictionaries

def compare_dictionaries(d1, d2, path=""):
    """
    Recursively compare two dicts and return a string listing all mismatches.
    If the return is empty, the dicts match exactly.
    """
    key_errs = []
    val_errs = []
    nested_errs = []

    # Check keys in d1
    for k in d1:
        current_path = f"{path}[{k!r}]"
        if k not in d2:
            key_errs.append(f"Key{current_path} missing in second dict")
        else:
            v1, v2 = d1[k], d2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                # recurse
                sub = compare_dictionaries(v1, v2, current_path)
                if sub:
                    nested_errs.append(sub)
            else:
                if v1 != v2:
                    val_errs.append(
                        f"Value at{current_path}: {v1!r} != {v2!r}"
                    )

    # Check keys in d2 that d1 lacks
    for k in d2:
        current_path = f"{path}[{k!r}]"
        if k not in d1:
            key_errs.append(f"Key{current_path} missing in first dict")

    # aggregate
    return key_errs + val_errs + nested_errs

#----------------------------------------------------------------------------
# Options which generate possible GuidanceVectorField configurations

LATENT_ARGS = [
         "ambient",
         {"seed" : random.randint(0,100), "dim_in" : random.randint(16,32), "dim_out" : random.randint(8, 16), "n_features" : random.randint(1,3)},
         {"net": "__REF__network", "hook_manager" : "__REF__hook_manager"} 
    ]

NOISE_GATE_ARGS = [
        {"type_gate" : "quadratic", "nu" : random.randint(0, 80)},
        {"type_gate" : "logistic", "nu" : random.randint(0, 80), "decay_rate" :  random.random()},
    ]

ARGS_NOISE_ARGS = ["edm"]

VECTORFIELD_ARGS = [
    {
        "noise_gate": gate, "args_noise": noise, 
        "scale": random.random(), 
        "features_template" : "__REF__features_template"
     }
    for gate, noise in itertools.product(
        NOISE_GATE_ARGS, ARGS_NOISE_ARGS
        )
    ]

PULLBACK_ARGS = [
        None,
        "jvp",
        # {step_size : ...}
    ]


ALL_CONFIGS = list(itertools.product(
        LATENT_ARGS,
        VECTORFIELD_ARGS,
        PULLBACK_ARGS,
        ARGS_NOISE_ARGS,
    ))


#----------------------------------------------------------------------------
# Assert config == load(save(config))

@pytest.mark.skip()
def test_json_save_and_load_equals_identity(config):
    assert config == load_from_json(save_to_json(config))

#----------------------------------------------------------------------------
# Assert config_out = create_config(create_gvf(config_in))

@pytest.mark.parametrize('args_latent, args_vectorfield, args_pullback, args_noise', ALL_CONFIGS)
def test_config_gvf_in_equals_config_gvf_out(args_latent, args_vectorfield, args_pullback, args_noise): 

    args_references = {}
    args_references["features_template"] = torch.zeros((1, 3, 2, 2))
    args_references["network"] = None
    args_references["hook_manager"] = None

    args_in = {
        "latent"       : args_latent,
        "vectorfield"  : args_vectorfield,
        "pullback"     : args_pullback,
        "noise"        : args_noise,
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
# TODO

