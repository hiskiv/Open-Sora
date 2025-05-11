num_frames = 50
resolution = "256"
aspect_ratio = "1:1"
fps = 24
frame_interval = 1

save_dir = "samples/v2v/debug"
multi_resolution = "STDiT2"
seed = 42
batch_size = 1
dtype = "bf16"

# reference_path = ["/users/xiaokangliu/projects/Open-Sora/assets/eval_images/995-frame_1.jpg"]

cond_type = "i2v_head"  # v2v_head, i2v_head, i2v_loop, or None
condition_frame_length = 5  # condition on the first n frames, latent size
use_sdedit = True
use_oscillation_guidance_for_text = True
use_oscillation_guidance_for_image = True

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/users/xiaokangliu/projects/Open-Sora/outputs/0005-STDiT3-XL-2/epoch193-global_step6000/model",
    # from_pretrained="/users/xiaokangliu/models/OpenSora-STDiT-v4-i2v/",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    kernel_size=(8, 8, -1),  # H W T
    use_spatial_rope=True,
    class_dropout_prob=0.0,
    force_huggingface=True,
)
vae = dict(
    type="OpenSoraVAE_V1_3",
    from_pretrained="/users/xiaokangliu/models/OpenSora-VAE-v1.3",
    z_channels=16,
    micro_batch_size=1,
    micro_batch_size_2d=4,
    micro_frame_size=17,
    use_tiled_conv3d=True,
    tile_size=4,
    normalization="video",
    temporal_overlap=True,
    force_huggingface=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.5,
    scale_image_weight=True,
    initial_image_scale=1.0,
)
image_cfg_scale = 5.0  # cfg, increase to force follow
aes = 7.0
flow = None
