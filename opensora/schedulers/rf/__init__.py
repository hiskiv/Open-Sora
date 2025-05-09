import math
import torch
from tqdm import tqdm
from typing import Optional, Tuple

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler
from .time_sampler import timestep_transform


def dynamic_thresholding(x, ratio=0.995, base=6.0):
    s = torch.quantile(x.abs().flatten(), ratio)
    s = max(s, base)
    x = x.clip(-s, s) * base / s
    return x


def get_oscillation_gs(guidance_scale, i, force_num=10):
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs


def fix_guidance_flaw(v_pred, pred_cond):
    # common diffision noise schedules
    std_cond = pred_cond.std(dim=list(range(1, pred_cond.dim())), keepdim=True)
    std_cfg = v_pred.std(dim=list(range(1, v_pred.dim())), keepdim=True)
    factor = std_cond / std_cfg
    factor = 0.7 * factor + (1 - 0.7)
    v_pred = v_pred * factor
    return v_pred


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        # time sampler
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        transform_scale=1.0,
        scale_temporal=True,
        # guidance
        use_oscillation_guidance=False,
        use_flaw_fix=False,
        scale_image_weight=False,
        initial_image_scale=1.0,
        force_num=10,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        self.scale_temporal = scale_temporal
        self.use_oscillation_guidance = use_oscillation_guidance
        self.use_flaw_fix = use_flaw_fix

        self.scale_image_weight = scale_image_weight
        self.initial_image_scale = initial_image_scale
        self.force_num = force_num

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            transform_scale=1.0,
            scale_temporal=scale_temporal,
            **kwargs,
        )

    def scale_weight(
        self,
        scale,
        init_scale,
        step_idx,
        total_steps,
        scale_method="decr",
    ):
        if scale_method == "incr":
            res_scale = torch.linspace(init_scale, scale, total_steps)[step_idx]
        elif scale_method == "decr":
            res_scale = torch.linspace(scale, init_scale, total_steps)[step_idx]
        else:
            raise NotImplementedError

        return res_scale

    def scale_temporal_weight(
        self,
        z,
        upper_scale,
        lower_scale,
    ):
        b, c, t, h, w = z.size()
        res_scale = torch.linspace(lower_scale, upper_scale, t)[None, None, :, None, None].repeat(b, c, 1, h, w)
        res_scale = res_scale.to(z.device, z.dtype)
        return res_scale

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,  # text
        image_cfg_scale=None,  # image
        progress=True,
        neg_prompts=None,
        z_cond=None,
        z_cond_mask=None,
        mask_index=None,
        use_sdedit=False,
        use_oscillation_guidance_for_text=None,
        use_oscillation_guidance_for_image=None,
    ):
        if use_oscillation_guidance_for_text is None:
            use_oscillation_guidance_for_text = self.use_oscillation_guidance
        if use_oscillation_guidance_for_image is None:
            use_oscillation_guidance_for_image = self.use_oscillation_guidance

        if z_cond is not None and z_cond_mask is not None:
            z_cond = z_cond * z_cond_mask

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        if image_cfg_scale is None:
            image_cfg_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(**text_encoder.tokenize_fn(prompts))
        y_null = text_encoder.null(n)  # [n, 1, 300, 4096] where n is batch size
        if neg_prompts is None:
            if mask_index is not None and len(mask_index) > 0:
                model_args["y"] = torch.cat([model_args["y"], y_null, y_null], 0)
            else:
                model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        else:
            y_null_model_args = text_encoder.encode(**text_encoder.tokenize_fn(neg_prompts))
            if mask_index is not None and len(mask_index) > 0:
                model_args["y"] = torch.cat([model_args["y"], y_null_model_args["y"], y_null_model_args["y"]], 0)
                model_args["mask"] = torch.cat(
                    [model_args["mask"], y_null_model_args["mask"], y_null_model_args["mask"]], 0
                )
            else:
                model_args["y"] = torch.cat([model_args["y"], y_null_model_args["y"]], 0)
                model_args["mask"] = torch.cat([model_args["mask"], y_null_model_args["mask"]], 0)

        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    additional_args,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    scale_temporal=self.scale_temporal,
                )
                for t in timesteps
            ]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)

        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:  # not for i2v and v2v, need to force mask=None
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            ## SDEdit
            if use_sdedit == True:
                if mask_index is not None and len(mask_index) > 0:  # use condition instead of noise for i2v and v2v
                    # NOTE: sdedit should add frames regardless of cfg to provide the same initial starting point
                    z_noise = self.scheduler.add_noise(z_cond, torch.randn_like(z_cond), t)
                    z = torch.where(z_cond_mask == 1, z_noise, z)
            text_gs = guidance_scale
            if use_oscillation_guidance_for_text:
                text_gs = get_oscillation_gs(guidance_scale, i, force_num=self.force_num)

            # classifier-free guidance
            if mask_index is not None and len(mask_index) > 0:
                # image_gs
                image_gs = image_cfg_scale
                if use_oscillation_guidance_for_image:
                    image_gs = get_oscillation_gs(image_cfg_scale, i, force_num=self.force_num)
                if self.scale_image_weight and image_gs > self.initial_image_scale:
                    upper_weight = self.scale_weight(
                        image_gs,
                        self.initial_image_scale,
                        i,
                        len(timesteps),
                        scale_method="decr",
                    )
                    image_gs = self.scale_temporal_weight(z, upper_weight, self.initial_image_scale)

                    # if type(image_gs) is not float:  # dev debug message
                    #     print(f"step {i}, image_gs:{image_gs[0,0,:,0,0]}")

                z_in = torch.cat([z, z, z], 0)
                t = torch.cat([t, t, t], 0)

                # cfg, text+image,  image only, nothing
                z_cond_in = torch.cat([z_cond, z_cond, torch.zeros_like(z_cond).to(z_cond.device).to(z_cond.dtype)], 0)
                z_cond_mask_in = torch.cat([z_cond_mask, z_cond_mask, z_cond_mask], 0)

                print(z_in.shape)
                print(t.shape)
                print(z_cond_in.shape)
                print(z_cond_mask_in.shape)
                print(y_null.shape)
                print(model_args)
                pred = model(
                    z_in,
                    t,
                    cond=z_cond_in,
                    cond_mask=z_cond_mask_in,
                    mask_index=mask_index,
                    y_null=y_null.repeat(
                        # z_in.shape[0], 1, 1, 1
                        3, 1, 1, 1
                    ),  # NOTE: this shall not contain neg prompt info, strictly null
                    **model_args,
                ).chunk(2, dim=1)[0]
                pred_cond, pred_uncond_text, pred_uncond_all = pred.chunk(3, dim=0)
                v_pred = (
                    pred_uncond_all
                    + image_gs * (pred_uncond_text - pred_uncond_all)
                    + text_gs * (pred_cond - pred_uncond_text)
                )
            else:
                z_in = torch.cat([z, z], 0)
                t = torch.cat([t, t], 0)
                pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + text_gs * (pred_cond - pred_uncond)
                if self.use_flaw_fix:
                    v_pred = fix_guidance_flaw(v_pred, pred_cond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def rflow_step_with_logprob(
        self,
        v_pred: torch.FloatTensor,           # velocity field from your model, same shape as `z`
        timestep: torch.Tensor,              # current t, shape (B,)
        sample: torch.FloatTensor,           # this is your z at time t, shape (B,C,H,W) or (B,D)
        dt: torch.Tensor,                    # Δt (already divided by self.num_timesteps), shape (B,)
        mask_t_upper: torch.Tensor,
        prev_sample: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        One stochastic rectified-flow step x_t → x_{t+Δt} with log-prob.

        Args:
          v_pred:      velocity field v(z_t,t) from your model
          timestep:    tensor of current timesteps t
          sample:      tensor z_t
          dt:          time increment Δt (normalized by num_timesteps)
          prev_sample: if given, uses this instead of sampling noise

        Returns:
          prev_sample: new samples z_{t+Δt}
          log_prob:    tensor of shape (B,) giving per-sample log-prob of the Gaussian step
        """

        # 1) build the mean: μ = z_t + v_pred * Δt
        #    reshape dt to broadcast over C,H,W (or D)
        shape = [sample.shape[0]] + [1] * (sample.ndim - 1)
        dt_b = dt.view(shape)
        mu = sample + v_pred * dt_b

        # 2) compute noise scale σ from your λ(t) schedule:
        #    σ² = 2 * λ(t) * Δt
        #    you need to have defined self.lambda_fn(t) → tensor (B,)
        # lambda_t = self.lambda_fn(timestep)          # shape (B,)
        lambda_t = timestep # linear lambda_fn
        sigma = torch.sqrt(2.0 * lambda_t.view(1, 1, 1, 1, 1).expand(*shape) * dt_b)

        # 3) sample (if not provided)
        if prev_sample is None:
            noise = torch.randn_like(sample)
            prev_sample = mu + sigma * noise

        # 4) compute per‑pixel log‑prob under N(μ,σ²):
        var = sigma * sigma
        log_prob = -((prev_sample - mu) ** 2) / (2.0 * var)
        log_prob = log_prob - torch.log(sigma) - 0.5 * math.log(2 * math.pi)

        # average over all but batch dimension → (B,)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob
    
    def sample_with_logprobs(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,  # text
        image_cfg_scale=None,  # image
        progress=True,
        neg_prompts=None,
        z_cond=None,
        z_cond_mask=None,
        mask_index=None,
        use_sdedit=False,
        use_oscillation_guidance_for_text=None,
        use_oscillation_guidance_for_image=None,
    ):
        if use_oscillation_guidance_for_text is None:
            use_oscillation_guidance_for_text = self.use_oscillation_guidance
        if use_oscillation_guidance_for_image is None:
            use_oscillation_guidance_for_image = self.use_oscillation_guidance

        if z_cond is not None and z_cond_mask is not None:
            z_cond = z_cond * z_cond_mask

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale
        if image_cfg_scale is None:
            image_cfg_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(**text_encoder.tokenize_fn(prompts))
        y_null = text_encoder.null(n)  # [n, 1, 300, 4096] where n is batch size
        self.model_args = model_args
        self.y_null = y_null
        if neg_prompts is None:
            if mask_index is not None and len(mask_index) > 0:
                model_args["y"] = torch.cat([model_args["y"], y_null, y_null], 0)
            else:
                model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        else:
            y_null_model_args = text_encoder.encode(**text_encoder.tokenize_fn(neg_prompts))
            if mask_index is not None and len(mask_index) > 0:
                model_args["y"] = torch.cat([model_args["y"], y_null_model_args["y"], y_null_model_args["y"]], 0)
                model_args["mask"] = torch.cat(
                    [model_args["mask"], y_null_model_args["mask"], y_null_model_args["mask"]], 0
                )
            else:
                model_args["y"] = torch.cat([model_args["y"], y_null_model_args["y"]], 0)
                model_args["mask"] = torch.cat([model_args["mask"], y_null_model_args["mask"]], 0)

        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    additional_args,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    scale_temporal=self.scale_temporal,
                )
                for t in timesteps
            ]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)

        all_latents = []
        all_next_latents = []
        all_log_probs = []
        gs = []
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:  # not for i2v and v2v, need to force mask=None
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            ## SDEdit
            if use_sdedit == True:
                if mask_index is not None and len(mask_index) > 0:  # use condition instead of noise for i2v and v2v
                    # NOTE: sdedit should add frames regardless of cfg to provide the same initial starting point
                    z_noise = self.scheduler.add_noise(z_cond, torch.randn_like(z_cond), t)
                    z = torch.where(z_cond_mask == 1, z_noise, z)
            text_gs = guidance_scale
            if use_oscillation_guidance_for_text:
                text_gs = get_oscillation_gs(guidance_scale, i, force_num=self.force_num)

            # classifier-free guidance
            if mask_index is not None and len(mask_index) > 0:
                # image_gs
                image_gs = image_cfg_scale
                if use_oscillation_guidance_for_image:
                    image_gs = get_oscillation_gs(image_cfg_scale, i, force_num=self.force_num)
                if self.scale_image_weight and image_gs > self.initial_image_scale:
                    upper_weight = self.scale_weight(
                        image_gs,
                        self.initial_image_scale,
                        i,
                        len(timesteps),
                        scale_method="decr",
                    )
                    image_gs = self.scale_temporal_weight(z, upper_weight, self.initial_image_scale)

                    # if type(image_gs) is not float:  # dev debug message
                    #     print(f"step {i}, image_gs:{image_gs[0,0,:,0,0]}")

                all_latents.append(z)

                z_in = torch.cat([z, z, z], 0)
                t = torch.cat([t, t, t], 0)

                # cfg, text+image,  image only, nothing
                z_cond_in = torch.cat([z_cond, z_cond, torch.zeros_like(z_cond).to(z_cond.device).to(z_cond.dtype)], 0)
                z_cond_mask_in = torch.cat([z_cond_mask, z_cond_mask, z_cond_mask], 0)

                pred = model(
                    z_in,
                    t,
                    cond=z_cond_in,
                    cond_mask=z_cond_mask_in,
                    mask_index=mask_index,
                    y_null=y_null.repeat(
                        3, 1, 1, 1
                    ),  # NOTE: this shall not contain neg prompt info, strictly null
                    **model_args,
                ).chunk(2, dim=1)[0]
                pred_cond, pred_uncond_text, pred_uncond_all = pred.chunk(3, dim=0)
                v_pred = (
                    pred_uncond_all
                    + image_gs * (pred_uncond_text - pred_uncond_all)
                    + text_gs * (pred_cond - pred_uncond_text)
                )

            else:
                z_in = torch.cat([z, z], 0)
                t = torch.cat([t, t], 0)
                pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + text_gs * (pred_cond - pred_uncond)
                if self.use_flaw_fix:
                    v_pred = fix_guidance_flaw(v_pred, pred_cond)

            # new stochastic update + log_prob
            dt = timesteps[i] - timesteps[i+1] if i < len(timesteps)-1 else timesteps[i]
            dt = dt / self.num_timesteps  # shape (B,)
            # z: [B, F, C, H, W], t, dt: [], t: [0, 1000], dt: [0, 1]
            z, step_log_prob = self.rflow_step_with_logprob(
                v_pred=v_pred,             # your already-computed velocity field
                timestep=t[0] / self.num_timesteps,                # the same t you passed into model(...)
                sample=z,                  # current z_t
                dt=dt,                     # normalized Δt
                mask_t_upper=mask_t_upper if mask is not None else None,
                prev_sample=None           # or pass in a tensor to bypass sampling
            )

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
            
            all_next_latents.append(z)
            all_log_probs.append(step_log_prob)
            gs.append((image_gs, text_gs))

        return z, timesteps, all_latents, all_next_latents, all_log_probs, z_cond, gs

    def sample_debug(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        neg_prompts=None,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(**text_encoder.tokenize_fn(prompts))

        # use "" as negative prompts if not provided
        if neg_prompts is None:
            y_null = text_encoder.null(n)
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        else:
            y_null_model_args = text_encoder.encode(neg_prompts)
            model_args["y"] = torch.cat([model_args["y"], y_null_model_args["y"]], 0)
            model_args["mask"] = torch.cat([model_args["mask"], y_null_model_args["mask"]], 0)

        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    additional_args,
                    scale=self.transform_scale,
                    num_timesteps=self.num_timesteps,
                    scale_temporal=self.scale_temporal,
                )
                for t in timesteps
            ]

        infos = []
        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            info = dict(i=i, t=t.cpu().item())
            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            info["z"] = z.clone()
            infos.append(info)

        return infos

    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        t=None,
        mask_index=None,
        text_uncond_prob=None,
        **kwargs,
    ):
        return self.scheduler.training_losses(
            model,
            x_start,
            model_kwargs,
            noise,
            mask,
            weights,
            t,
            mask_index=mask_index,
            text_uncond_prob=text_uncond_prob,
            **kwargs,
        )
    
    def RL_training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        noise=None,
        mask=None,
        weights=None,
        z_cond=None,
        t=None,
        timesteps=None,
        mask_index=None,
        text_uncond_prob=None,
        gs=None,
        next_latents=None,
        **kwargs,
    ):
        v_pred = self.scheduler.RL_training_losses(
            model=model,
            x_start=x_start,
            model_kwargs=self.model_args,
            noise=noise,
            mask=mask,
            weights=weights,
            z_cond=z_cond,
            t=timesteps[t],
            mask_index=mask_index,
            text_uncond_prob=text_uncond_prob,
            image_gs=gs[0],
            text_gs=gs[1],
            y_null=self.y_null,
            **kwargs,
        )

        dt = timesteps[t] - timesteps[t+1] if t < len(timesteps)-1 else timesteps[t]
        dt = dt / self.num_timesteps  # shape (B,)
        # z: [B, F, C, H, W], t, dt: [], t: [0, 1000], dt: [0, 1]
        _, step_log_prob = self.rflow_step_with_logprob(
            v_pred=v_pred,
            timestep=timesteps[t][0] / self.num_timesteps,                # the same passed into model(...)
            sample=x_start,                  # current z_t
            dt=dt,                     # normalized Δt
            mask_t_upper=None,
            prev_sample=next_latents
        )

        return step_log_prob
