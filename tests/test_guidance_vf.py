import pytest
import torch
from mylib.diffusion import ConfigGuidanceVF, create_guidance_vf


def test_weight_thresholding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    template = torch.randn(1, 3, 2, 2, device=device, dtype=dtype)
    
    prms = ConfigGuidanceVF(
        type_latent="pixel",
        scale_template_score=1.0,
        v_0=40.0,
        decay_rate=1.0,
        threshold_weight=0.5,
    )
    
    # At v_0 =t and scale_template_score = 1.0 weight is 0.5
    # Therefore any weight with t under that thresdhold will be zero
    vf = create_guidance_vf(prms, template, verbose=False)
    x = torch.randn_like(template)
    t =  torch.rand(1) + 38.5

    score = vf(x, t)

    assert vf.threshold_weight == 0.5
    assert vf.history_weight[-1] < 0.5 
    assert vf.history_apply_score[-1] == False
    assert torch.allclose(score, torch.zeros_like(score)), (
        f"Score mismatch:\n"
        f"Expected: all zeros (shape {tuple(score.shape)})\n"
        f"Actual:   {score}"
    )


def test_time_thresholding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    template = torch.randn(1, 3, 2, 2, device=device, dtype=dtype)
    
    prms = ConfigGuidanceVF(
        type_latent="pixel",
        scale_template_score=1.0,
        v_0=40.0,
        decay_rate=1.0,
        threshold_time_min=1.0,
        threshold_time_max=2.0
    )
    
    vf = create_guidance_vf(prms, template, verbose=False)
    x = torch.randn_like(template)
    
    # Test t below min threshold
    t_below = 0.5
    score_below = vf(x, torch.tensor(t_below, dtype=dtype, device=device))
    assert torch.allclose(score_below, torch.zeros_like(score_below))
    assert vf.history_apply_score[-1] == False
    
    # Test t above max threshold
    t_above = 3.0
    score_above = vf(x, torch.tensor(t_above, dtype=dtype, device=device))
    assert torch.allclose(score_above, torch.zeros_like(score_above))
    assert vf.history_apply_score[-1] == False
    
    # Test t within thresholds
    t_within = 1.5
    score_within = vf(x, torch.tensor(t_within, dtype=dtype, device=device))
    assert vf.history_apply_score[-1] == True

