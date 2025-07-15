import pytest
import torch
from mylib.gvf import ConfigGVFAmbient, ConfigGVFUnet, ConfigGVFUnetAttention,  create_vf, AttentionMixture, BuidlerUNetGVF, BuilderUNetAttentionGVF
from mylib.diffusion import load_templates, load_templates_batch, ConfigSimulation
import dnnlib
import pickle
from training.networks import EDMPrecond, HookManager
from torch_utils import misc

#-------Thresholding-------


@pytest.mark.skip
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


@pytest.mark.skip
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

#------- Attention ------- 

@pytest.fixture
def setup_attention():
    torch.manual_seed(42)
    N = 5  
    D = 3   
    batch_size = 4

    means = torch.randn(N, D)
    stds = torch.ones(N) * 0.5
    mix_weights = torch.ones(N) # TODO: all tests assume that all components are equally weighted
    mix_weights /= mix_weights.sum()  
    std_noise = 0.1
    x = torch.randn(batch_size, means.shape[1])

    return means, stds, mix_weights, std_noise, x 

@pytest.mark.skip
def test_attention_sum_to_one(setup_attention):
    means, stds, mix_weights, std_noise, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    assert torch.allclose(attn.sum(axis=-1), torch.tensor(1.0), atol=1e-5), "Attention weights do not sum to 1"

@pytest.mark.skip
def test_attention_less_than_one(setup_attention):
    means, stds, mix_weights, std_noise, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    assert torch.all(attn < 1.0), "Some attention weights are not strictly less than 1"

@pytest.mark.skip
def test_attention_highest_for_closest_mean(setup_attention):
    means, _, mix_weights, std_noise, x = setup_attention
    stds = torch.ones(means.shape[0]) * 0.5 # make sure stds are equal
    attention_fn = AttentionMixture(means, stds, mix_weights)
    attn = attention_fn(x, std_noise)

    # Identify index of closest mean
    diff = torch.norm(means.unsqueeze(0) - x.unsqueeze(1), dim=2)
    closest_idx = diff.argmin(axis=-1)

    # Check that the closest mean gets the highest attention
    max_idx = attn.argmax(axis=-1)
    assert torch.all(max_idx == closest_idx), (
        f"Attention max index {max_idx} != closest mean index {closest_idx}"
    )

@pytest.mark.skip
def test_attention_becomes_uniform_as_noise_increases(setup_attention):
    means, stds, mix_weights, _, x = setup_attention
    attention_fn = AttentionMixture(means, stds, mix_weights)
    pass
    # TODO: calculate KL divergence as noise increase and ensure it's monotonic

    raise NotImplementedError()

@pytest.mark.skip
def test_batched_attention_of_singles_is_ones():
    raise NotImplementedError()

@pytest.fixture(scope="module")
def setup_edm_model():
    MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    with dnnlib.util.open_url(f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl') as f:
        net_old = pickle.load(f)['ema'].to(device)

    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(device)
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)
    return net, device, dtype


def test_unet_latents_match_unet_fwd(setup_edm_model):
    # Setup network, templates and config
    net, device, dtype = setup_edm_model
    templates = load_templates_batch(["data/data/cat_1.jpg"], device=device, dtype=dtype) 
    sigma = torch.tensor(10).to(device).to(dtype)
    cfgs = {
        "unet-attn" : (BuilderUNetAttentionGVF, ConfigGVFUnetAttention( type_eval = "numdiff", idxs = tuple(range(6, 9)), vf_latent = ConfigGVFAmbient())),
        # "unet-skips" :(BuidlerUNetGVF, ConfigGVFUnet(type_eval = "numdiff", idx_skips = (15, ), vf_latent = ConfigGVFAmbient())),
    }

    
    for name, (Builder, cfg) in cfgs.items():
        # Run U-Net pass
        y = net(templates, sigma)
        y  = (y * 127.5 + 128) / 255
        # Setup U-net builder builder and ensure it has same output as forward
        builder = Builder(cfg, templates, device=device, dtype=dtype, net=net)
        builder._setup_latents()

        y_test = builder.latent_inv_fn(builder.latent_fn(templates))
        y_test  = (y_test * 127.5 + 128) / 255
        
        if not torch.equal(y, y_test):
            import matplotlib.pyplot as plt
            from einops import rearrange
            fig, axes = plt.subplots(2, 1)
            axes[0].imshow(rearrange(y.detach().cpu(), "1 C H W -> H W C"))
            axes[1].imshow(rearrange(y_test.detach().cpu(), "1 C H W -> H W C"))
            plt.show()

            pytest.fail(f"Latents {name} do not match fwd pass")

@torch.no_grad()
def test_hook():
    MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

    cfg = ConfigGVFUnetAttention(
        type_eval = "numdiff",
        idxs = tuple(range(4, 16)),
        vf_latent = ConfigGVFAmbient()
    )

    MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    device= "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    templates = load_templates_batch(["data/data/cat_1.jpg"], device=device, dtype=dtype) 
    network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl'

    with dnnlib.util.open_url(network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)

    # Load two images
    x_1 = load_templates("data/data/cat_1.jpg", device=device, dtype=dtype)
    x_2 = load_templates("data/data/cat_2.jpg", device=device, dtype=dtype)
    sigma = torch.tensor(10).to(device).to(dtype)

    # Register UNet Block 4 - 16 
    hook_manager = HookManager()

    blocks_with_attention = []
    for name, block in net.model.enc.items():
        # Check if block uses attention
        if getattr(block, "num_heads", 0) > 0:
            # Log the name 
            blocks_with_attention.append(name)
            hook_manager.register(name)
    assert len(blocks_with_attention) == 9
    net.hook_manager = hook_manager

    # Enable saving each run
    hook_manager.save_current_run = True
    # Get attention for first image and save weights for later
    hook_manager.save_blocks = True
    net(x_1, sigma)
    attn_1  = hook_manager.load_current()
    hook_manager.reset_current()
    hook_manager.save_blocks = False
    # Get attention for second image
    net(x_2, sigma)
    attn_2  = hook_manager.load_current()
    hook_manager.reset_current()
    # Run second image again
    hook_manager.load_blocks = True
    net(x_2, sigma)
    attn_combined = hook_manager.load_current()
    hook_manager.reset_current()

    # Assert attention for two images are not the same
    for idx, (a, b) in enumerate(zip(attn_1, attn_2)):
        assert not torch.equal(a, b), f"Attention matches at index = {idx}"

    # Assert 1-3 match Image 2 and 4-9 match Image 1
    for idx, (a, b) in enumerate(zip(attn_combined, attn_2[:3] + attn_1[3::])):
        assert torch.equal(a, b), f"Attention doesn't match at index = {idx}"


def test_forward_capture():
    # Setup
    MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    device= "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    templates = load_templates_batch(["data/data/cat_1.jpg"], device=device, dtype=dtype) 
    network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl'

    with dnnlib.util.open_url(network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)
    
    # Test data
    x = torch.randn(1, 3, 64, 64).to(device=device, dtype=dtype)
    sigma = torch.tensor([0.5]).to(device=device, dtype=dtype)
    labels = None
    
    # Execute
    hook_manager = HookManager()
    hook_manager.save_fwd = True
    net.hook_manager = hook_manager
    net(x, sigma, class_labels=labels, force_fp32=True, augment_labels=None)
    
    # Verify
    fwd_vars = hook_manager.fwd_vars
    assert torch.equal(fwd_vars.x, x)
    assert torch.equal(fwd_vars.sigma, sigma)
    assert fwd_vars.class_labels == labels
    assert fwd_vars.force_fp32 is True
    assert fwd_vars.model_kwargs["augment_labels"] == None
    
    # Test reset
    hook_manager.reset_fwd()
    assert hook_manager.fwd_vars is None
    
    
