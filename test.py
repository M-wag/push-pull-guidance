### Attention Mechanism### 
import torch
import pytest
from example import AttentionMixture  # replace with actual module name if needed

@pytest.fixture
def setup_attention():
    torch.manual_seed(42)
    N = 5  
    D = 3   

    means = torch.randn(N, D)
    stds = torch.ones(N) * 0.5
    mix_weights = torch.ones(N) # TODO: all tests assume that all components are equally weighted
    mix_weights /= mix_weights.sum()  
    std_noise = 0.1
    x = torch.randn(means.shape[1])

    return means, stds, mix_weights, std_noise, x 

def test_attention_sum_to_one(setup_attention):
    means, stds, mix_weights, std_noise, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5), "Attention weights do not sum to 1"

def test_attention_less_than_one(setup_attention):
    means, stds, mix_weights, std_noise, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    assert torch.all(attn < 1.0), "Some attention weights are not strictly less than 1"

def test_attention_highest_for_closest_mean(setup_attention):
    means, _, mix_weights, std_noise, x = setup_attention
    stds = torch.ones(means.shape[0]) * 0.5 # make sure stds are equal
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    # Identify index of closest mean
    distances = torch.norm(means - x.unsqueeze(0), dim=1)
    closest_idx = distances.argmin()

    # Check that the closest mean gets the highest attention
    max_idx = attn.argmax()
    assert max_idx == closest_idx, (
        f"Attention max index {max_idx} != closest mean index {closest_idx}"
    )

def test_attention_becomes_uniform_as_noise_increase(setup_attention):
    means, stds, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)

    # TODO: calculate KL divergence as T increase and ensure it's monotonic
