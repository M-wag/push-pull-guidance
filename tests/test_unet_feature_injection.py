import copy
import torch
import pytest
import random
from contextlib import contextmanager
from training.networks import EDMPrecond, UNetBlock, InjectionManager

#----------------------------------------------------------------------------
# Fixtures for AttentionBlock and EDMPrecond

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
# Helper function to deal with copying of when type is ambigious

def safe_copy(obj):
    """
    Safely copy an object that could be a tensor, list, or other type.
    Returns a copy that preserves the original structure and values.
    """
    if isinstance(obj, torch.Tensor):
        # Clone tensors to preserve their values
        return obj.clone()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # Recursively copy list elements
        return type(obj)(safe_copy(item) for item in obj)
    elif isinstance(obj, dict):
        # Recursively copy dictionary values
        return {key: safe_copy(value) for key, value in obj.items()}
    else:
        # For other types, use standard copy
        return copy.copy(obj)

#----------------------------------------------------------------------------
# Context manager for tracking calls 

@contextmanager
def track_method(obj, method_name):
    """
    Track calls to a specific method on an object.
    Yields a tracker object with input and output information.
    """
    
    class Tracker():
        def __init__(self):
            self.called = False
            self.call_count = 0
            self.inputs = []
            self.outputs = []

        @property
        def last_args(self):
            return self.inputs[-1][0]

    tracker = Tracker()
    original_method = getattr(obj, method_name)

    def tracked_method(*args, **kwargs):
        tracker.called = True
        tracker.call_count += 1
        tracker.inputs.append((safe_copy(args), safe_copy(kwargs)))
        result = original_method(*args, **kwargs)
        tracker.outputs.append(result)
        return result

    try: 
        setattr(obj, method_name, tracked_method)
        yield tracker 
    finally:
        setattr(obj, method_name, original_method)


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
    spy = mocker.spy(attention_block, "compute_attention") 
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
# Test integration for attention injection between EDMPrecond and UNetBlocks

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
    spies = [mocker.spy(net.model.enc[name], "compute_attention") for name in names_registered]

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


def test_save_load_skips_only_when_registered():
    pass

#----------------------------------------------------------------------------
# When saving then loading make sure attention matches

def test_skips(edm_net):
    # Setup network and pick random skips to register
    net = edm_net
    net.set_injection_manager(InjectionManager())
    idxs = sorted(random.sample(range(0, net.num_skips), k=max(net.num_skips // 3, 1)))
    for idx in idxs:
        net.register_injection((f"skip_{idx}", "skip"))
    
    # Random inputs
    a = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    b = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    t = torch.rand(1) * 80

    with track_method(net.model, "decoder") as tracker:
        # Run the model and save track skip connections.
        with net.injection_mode(save_features=True):
            net(a, t)
        skips_a = tracker.last_args[0]
    
        # Run the model again with different values
        _ = net(b, t)
        skips_b = tracker.last_args[0]

        for index in idxs:
            skip_a = skips_a[index]
            skip_b = skips_b[index]
            
            # Verify that this specific skip connection differs between runs
            assert not torch.equal(skip_a, skip_b), \
                f"Skip connection at index {index} was identical between runs. " \
                "This suggests injection loading may be occurring when it shouldn't be"

        # Run the model again with differnet values, but with loading on
        with net.injection_mode(load_features=True):
            net(b, t)
        skips_b_loaded = tracker.last_args[0]

        assert len(skips_a) == len(skips_b)

        for index in range(len(skips_a)):
            if index in idxs:
                skip = skips_a[index]
            else:
                skip = skips_b[index]

            skip_b = skips_b_loaded[index]
            
            # Verify that this specific skip connection is identical between runs
            assert torch.equal(skip, skip_b), \
                f"Indices : {idxs} \n" \
                f"Skip connection at index {index} differs between runs. " \
                "This suggests injection loading may not be working correctly."

#----------------------------------------------------------------------------
# Make sure num skips matches len of sis 

def test_num_skips_match_length_skips(edm_net):
    # Initialize model and pick random inputs
    net = edm_net
    a = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    t = torch.rand(1) * 80

    with track_method(net.model, "decoder") as tracker:
        out = net(a, t)

    skips = tracker.last_args[0]

    assert net.num_skips == len(skips), \
        f"Length of encoder output ({len(skips)}) should match net.num_skips property ({net.num_skips})"

#----------------------------------------------------------------------------
# decoder(encoder(x, t), t) == net(x ,t)

def test_decoder_encoder_pass_matches_net_forward(edm_net):
    # Initialize model and pick random inputs
    net = edm_net
    x = torch.randn(2, net.img_channels, net.img_resolution, net.img_resolution)
    t = torch.rand(1) * 80

    skips = net.encoder(x, t)
    y = net.decoder(x, skips, t)

    assert torch.equal(y, net(x, t))

