import pytest
import torch
import dnnlib
import pickle

from mylib.gvf_2 import AttentionMixture


@pytest.fixture
def setup_attention():
    # torch.manual_seed(42)
    N = 5  
    D = 2   
    B = 4

    means = torch.randn(B, N, D)
    std = 1
    mix_weights = torch.ones(N) # TODO: all tests assume that all components are equally weighted
    mix_weights /= mix_weights.sum()  
    T = 0.1
    x = torch.randn(B, D)

    return means, std, mix_weights, T, x 

def test_attention_sum_to_one(setup_attention):
    means, std, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, std, mix_weights)
    attn = attention_fn(x)

    assert torch.allclose(attn.sum(axis=-1), torch.tensor(1.0), atol=1e-5), "Attention weights do not sum to 1"

def test_attention_between_zero_and_one(setup_attention):
    means, std, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, std)
    attn = attention_fn(x)

    assert torch.all((attn < 1.0) & (attn >= 0.0)), "Some attention weights are not strictly between 0 and 1"

def test_attention_highest_for_closest_mean(setup_attention):
    means, std, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, std, mix_weights)
    attn = attention_fn(x)

    # Identify index of closest mean
    diff = torch.norm(means - x.unsqueeze(1), dim=-1)
    closest_idx = diff.argmin(axis=-1)

    # Check that the closest mean gets the highest attention
    max_idx = attn.argmax(axis=-1)
    assert torch.all(max_idx == closest_idx), (
        f"Attention max index {max_idx} != closest mean index {closest_idx}"
    )

