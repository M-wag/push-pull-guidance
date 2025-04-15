### Attention Mechanism### 
import torch
import pytest
from lib import PullBackGradient, LinearPullBackGradient

@pytest.fixture
def setup_attention():
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

def test_attention_becomes_uniform_as_noise_increases(setup_attention):
    means, stds, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    raise NotImplementedError
    # TODO: calculate KL divergence as noise increase and ensure it's monotonic

def test_batched_attention_of_singles_is_ones():
    raise NotImplementedError

### Pull Back Gradients

@pytest.fixture
def setup_pullback_gradients():
    T_max = 80
    template = torch.rand((1, 3, 64, 64))
    v_0 = torch.rand(1) * T_max
    decay_rate = torch.rand(1) * 0.2 + 0.9
    dim_features = 64 
    return template, v_0, decay_rate, dim_features, T_max

def test_pb_grad_linear_pb_grad_equivalence(setup_pullback_gradients):
    template, v_0, decay_rate, dim_features, T_max  = setup_pullback_gradients
    dim_template = torch.prod(torch.tensor(template.shape)[1::])

    # Init random feature matrix
    A = torch.rand((dim_features, dim_template))
    A_inv = torch.linalg.pinv(A)
    latent = lambda x :  (x @ A.T)
    latent_inv = lambda x :  (x @ A_inv.T)

    # Init linear and general pullback gradient
    pb_grad = PullBackGradient(template, v_0, decay_rate, latent, latent_inv, flatten_input=True) 
    lin_pb_grad = LinearPullBackGradient(template, v_0, decay_rate, A, flatten_input=True) 

    # Random time and batch
    t = torch.rand(1) * T_max
    x = torch.randn((8, *template.shape))

    grad_pb = pb_grad(x, t)
    grad_lin = lin_pb_grad(x, t)
    assert torch.all(torch.isclose(grad_pb, grad_lin)), \
            "Gradients for the (general) PullBackGradient and LinearPullBackGradient when accepting the same feature matrix" 


if __name__ == "__main__":
    pass


