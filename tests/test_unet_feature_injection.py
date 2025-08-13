import torch
import pytest
import random

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


@pytest.fixture()
def edm_net():
    return EDMPrecond(img_resolution=16, img_channels=3)

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
# Test integration between EDMPrecond and UNetBlocks

def test_edm_properly_manges_unet_feature_injection(edm_net, mocker):
    net = edm_net
    net.set_injection_manager(InjectionManager())
    a = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    b = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    t = torch.rand(1) * 80

    # Get all blocks with name 
    names_blocks_with_attention = []
    for name in net.names_unet_blocks["enc"]:
        num_heads = getattr(net.model.enc[name], "num_heads", 0)
        if num_heads > 0:
            names_blocks_with_attention.append(name)


    # Take a random subset of the names
    names_registered = random.sample(names_blocks_with_attention, random.randint(1, len(names_blocks_with_attention)))

    # Register blocks
    net.register_injection([(name, "attention") for name in names_registered])

    # Track registerd Unet Block
    spies = [mocker.spy(net.model.enc[name], "get_attention") for name in names_registered]

    # Run network and save values
    net.enable_injection_saving(True)
    _ = net(a, t)
    net.enable_injection_saving(False)
    attns_a = [spy.spy_return.detach().clone() for spy in spies]

    # Run network with different values
    _ = net(b, t)
    attns_b = [spy.spy_return.detach().clone() for spy in spies]

    # Verify attention outputs changed with different input
    mismatches = []
    for name, attn_a, attn_b in zip(names_registered, attns_a, attns_b):
        if torch.allclose(attn_a, attn_b,):
            mismatches.append(name)
    
    if mismatches:
        pytest.fail(f"Attention didn't change for blocks: {mismatches}")

    # Run network with different values, but loading oenable
    net.enable_injection_loading(True)
    _ = net(b, t)
    attns_b_loaded = [spy.spy_return.detach().clone() for spy in spies]

    # Verify loaded attention matches first run's attention
    mismatches = []
    for name, attn_a, attn_b_loaded in zip(names_registered, attns_a, attns_b_loaded):
        if not torch.allclose(attn_a, attn_b_loaded):
            mismatches.append(name)
    
    if mismatches:
        pytest.fail(f"Loaded attention mismatch in blocks: {mismatches}")

def test_edm_identical_when_injection_manager(edm_net):
    net = edm_net
    x = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    t = torch.rand(1) * 80

    net.set_injection_manager(None)
    out_a = net(x, t)

    net.set_injection_manager(InjectionManager())
    out_b = net(x, t)

    assert torch.allclose(out_a, out_b), \
            "Output of EDM network should not be changed when having an empty InjectionManager"

    
