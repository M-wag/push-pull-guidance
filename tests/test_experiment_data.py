import os 
import torch
import dnnlib.util as util


#----------------------------------------------------------------------------
# Let there be a dict appended to *.json .
# Ensure dict = convert(read(*.json)) 

def test_():
    path_records = "tests/tests.json"
    # delete path if it exists
    if os.path.exists(path_records):
        os.remove(path_records)

    # example config
    config_og = {
        "sampler_kwargs" : {
            "sigma_min"         : 0.002 , 
            "sigma_max"         : 80, 
            "rho"               : 7, 
            "S_noise"           : 1.0,
            "S_churn"           : 40.0,
            "S_min"             : 0.05, 
            "S_max"             : 50,
            "dtype"             : torch.float32,
            "correct_rgb"       : False,
            "num_steps"         : 32,
            "apply_2nd_order"   : True,
        }, 

        "gvf_kwargs" : None,

        "generate_kwargs" : {
            "ddim_inversion"        : False,
            "live_editing"          : False,
            "use_noisy_examples"    : False,
            "example_idx_range"     : None,
        },

        "gradient_kwargs" : {
            "scale_model_score" : 1.0, 
        },
    }

    # make entry with append to records
    entry_old =  dict(run_id=0, **config_og)
    util.append_to_records(path_records, entry_old)

    # loads from records
    run_id = util.get_last_run_id_records(path_records)
    entry = util.get_entry_from_records(path_records, run_id=run_id)
    # convert to config 
    config_recon = util.convert_entry_to_config(entry)

    # Remove path after test
    os.remove(path_records)

    # check identity
    assert config_og == config_recon, \
            "Reconstructed config does not match old config"



