### Attention Mechanism### 
import torch
import pytest
from lib import PullBackGradient, LinearPullBackGradient

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
    assert torch.all(torch.isclose(grad_pb, grad_lin, atol=1e-6)), \
            "Gradients for the (general) PullBackGradient and LinearPullBackGradient when accepting the same feature matrix" 
if __name__ == "__main__":
    pass


