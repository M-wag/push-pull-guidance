import pytest
import torch

from mylib.gvf import UNetLatentBuilder
from dnnlib.util import to_easydict

@pytest.fixture()
def unet_builder():
    return UNetLatentBuilder(
            args = to_easydict({'net' : None, "attribute" : "attention", "index" : []}),
            args_inv = None,
            shp = None,
            device = "cpu",
            dtype = torch.float32,
    )
    
#----------------------------------------------------------------------------
# Ensure that pad and concat when undone matches original input

def test_pad_and_concat_returns_og_tensor_when_undone(unet_builder):
    tensors = [
        torch.rand(2, 3, 16, 16),
        torch.rand(2, 9, 8, 8),
        torch.rand(2, 27, 4, 4),

    ] 
    z = unet_builder.zero_padding_and_concat(tensors)
    tensors_recon = unet_builder.undo_zero_padding_and_concat(z)

    for og, recon in zip(tensors, tensors_recon):
        assert og.shape == recon.shape
        assert torch.equal(og, recon)


