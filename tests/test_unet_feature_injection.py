import torch
import pytest

from training.networks import EDMPrecond, UNetBlock, InjectionManager

@pytest.fixture()
def attention_block():
    return UNetBlock(
        in_channels = 4,
        out_channels = 4,
        emb_channels = 5,
        num_heads = 1,
        attention = True,
        name = "test_block",
    )

#----------------------------------------------------------------------------
# Conditions under which no saving or loading should be done.
# Tested by seeing if the qkv(), save() and load() methods of UNetBlock are called.

def test_no_manager_no_save_load(attention_block, mocker):
    attention_block.injection_manager = None
    x = torch.randn(2, 4, 2, 2)
    emb = torch.rand(2, 5)

    # Track whethr qkv.forward(), save() and load() are run
    spy_qkv_forward = mocker.spy(attention_block.qkv, "forward")
    spy_save = mocker.spy(attention_block, "save")
    spy_load = mocker.spy(attention_block, "load")

    # Run block
    out = attention_block(x, emb)
    spy_qkv_forward.assert_called_once()
    spy_save.assert_not_called()
    spy_load.assert_not_called()


def test_save_load_only_when_registered(attention_block, mocker):
    #  Injection manager but name not registered
    injection_manager = InjectionManager()
    attention_block.injection_manager = injection_manager
    x = torch.randn(2, 4, 2, 2)
    emb = torch.rand(2, 5)

    # Track whether qkv.forward(), save() and load() are run
    spy_qkv_forward = mocker.spy(attention_block.qkv, "forward")
    spy_save = mocker.spy(attention_block, "save")
    spy_load = mocker.spy(attention_block, "load")

    # Run block 
    out = attention_block(x, emb)

    spy_qkv_forward.assert_called()
    spy_save.assert_not_called()
    spy_load.assert_not_called()
    mocker.resetall()

    # Register Different Block
    injection_manager.register(("not_test_block", "attention"))
    injection_manager.set_saving(True)
    out = attention_block(x, emb)

    spy_qkv_forward.assert_called()
    spy_save.assert_not_called()
    spy_load.assert_not_called()
    mocker.resetall()

    # Register Block
    injection_manager.register(("test_block", "attention"))
    out = attention_block(x, emb)
    
    spy_qkv_forward.assert_called()
    spy_save.assert_called()
    spy_load.assert_not_called()
    mocker.resetall()

    # Turn off saving and turn on loading
    injection_manager.set_saving(False)
    injection_manager.set_loading(True)
    out = attention_block(x, emb)

    spy_qkv_forward.assert_not_called()
    spy_save.assert_not_called()
    spy_load.assert_called()

#----------------------------------------------------------------------------
# When saving then loading make sure attention matches

def test_attention_equal_when_save_then_load(attention_block, mocker):
    injection_manager = InjectionManager()
    attention_block.injection_manager = injection_manager
    a = torch.randn(2, 4, 2, 2)
    b = torch.randn(2, 4, 2, 2)
    emb = torch.rand(2, 5)

    injection_manager.register(("test_block", "attention"))

    # Run the model and save track attention layer.
    injection_manager.set_saving(True)
    spy = mocker.spy(attention_block, "get_attention") 
    _ = attention_block(a, emb)
    injection_manager.set_saving(False)
    attn_a = spy.spy_return.detach().clone()

    # Run the model again with differnet values
    _ = attention_block(b, emb)
    attn_b = spy.spy_return.detach().clone()

    assert not torch.all(attn_a == attn_b), \
            "Loaded attention should be different to previous run when injection_manager not loading"

    # Run the model again with differnet values, but with loading on
    injection_manager.set_loading(True)
    _ = attention_block(b, emb)
    attn_b_loaded = spy.spy_return.detach().clone()

    assert torch.all(attn_a == attn_b_loaded), \
            "Loaded attention should identitical first run when injection_manager is loading"


#----------------------------------------------------------------------------
# Test integration between EDMPrecond and DhariwalUNet

def test_edm():
    # Get all p
    net = EDMPrecond()

    assert False
