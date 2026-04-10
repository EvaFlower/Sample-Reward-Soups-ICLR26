"""
Stable Diffusion 3 Pipeline with Preference Guidance and Demon SOUP with Weight Sampling.

Adapted from SD 1.5 DDIM-based implementation to SD3 Flow Matching.
This version faithfully preserves the original algorithm including anchor-based sampling.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

# from .sd3_seperate_demon import (
#     flow_match_step_fetch_x0,
#     flow_match_step_fetch_x_t_1,
# )
from .sd3_sde_with_logprob import sde_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusion3Pipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    max_sequence_length: int = 256,
    # Custom parameters
    is_noise_images: bool = False,
    pref_function: Optional[List] = None,
    opt_func_name: Optional[List[str]] = None,
    pref_guide: float = 1.0,
    divert_start_step: int = 0,
    divert_end_step: int = 28,
    soup_start_step: int = 0,
    soup_end_step: int = 0,
    extra_info: Optional[Dict] = None,
    inner_steps: int = 1,
    accelerator = None,
    method: Optional[str] = None,
    weights: Optional[np.ndarray] = None,
    reward_norm: str = 'std',
    sym_noise_flag: bool = True,
    cor_flag: bool = True,
    query_size: int = 30,
    noise_level: float = 0.7,
    sde_type: Optional[str] = 'sde'
):
    """
    SD3 Pipeline with SOUP preference guidance.
    Preserves exact logic from original SD 1.5 implementation.
    """
    
    # 0. Default height and width
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    # 1. Check inputs
    self.check_inputs(
        prompt, prompt_2, prompt_3, height, width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )
    
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False
    
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    
    # 3. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, None, None
    )
    
    # 4. Encode input prompt
    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) 
        if self.joint_attention_kwargs is not None else None
    )
    
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    log_prompt_embeds = prompt_embeds.clone()
    log_pooled_prompt_embeds = pooled_prompt_embeds.clone()
    
    # Query configuration - matching original
    query_batch_size = 10
    assert query_size % query_batch_size == 0, \
        f"query_size ({query_size}) must be divisible by query_batch_size ({query_batch_size})"
    
    # Setup prompt embeds for CFG
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, log_prompt_embeds])
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, log_pooled_prompt_embeds])
        
        # For querying next timestep
        prev_prompt_embeds = torch.cat([
            negative_prompt_embeds.repeat(query_batch_size, 1, 1),
            log_prompt_embeds.repeat(query_batch_size, 1, 1),
        ])
        prev_pooled_prompt_embeds = torch.cat([
            negative_pooled_prompt_embeds.repeat(query_batch_size, 1),
            log_pooled_prompt_embeds.repeat(query_batch_size, 1),
        ])
    else:
        prev_prompt_embeds = log_prompt_embeds.repeat(query_batch_size, 1, 1)
        prev_pooled_prompt_embeds = log_pooled_prompt_embeds.repeat(query_batch_size, 1)
    
    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    # 6. Initialize
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    preference_score_logs = []
    compress_target = -10
    
    # If starting divergent sampling from step 0
    if divert_start_step == 0:
        latents = latents.repeat(len(weights), 1, 1, 1)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([
                negative_prompt_embeds.repeat(len(weights), 1, 1),
                log_prompt_embeds.repeat(len(weights), 1, 1),
            ])
            pooled_prompt_embeds = torch.cat([
                negative_pooled_prompt_embeds.repeat(len(weights), 1),
                log_pooled_prompt_embeds.repeat(len(weights), 1),
            ])
    
    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            
            for j in range(inner_steps):
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                
                # Broadcast timestep to batch
                timestep = t.expand(latent_model_input.shape[0])
                
                # Predict velocity
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if divert_end_step > i >= divert_start_step:
                    # PREFERENCE GUIDANCE REGION
                    
                    if soup_start_step <= i < soup_end_step:
                        # ============ SOUP SAMPLING WITH ANCHOR WEIGHTS ============
                        
                        prev_latents = [[] for _ in range(len(weights))]
                        
                        # Initialize noise (symmetric if specified)
                        if opt_func_name[1] == 'compress' and sym_noise_flag:
                            latent_shape = (query_size//2,) + latents.shape[1:]
                            tmp = torch.randn(latent_shape, device=device)
                            all_prev_latents_noise = torch.cat([tmp, -tmp], dim=0)
                        else:
                            all_prev_latents_noise = [None for _ in range(query_size)]
                        
                        # Anchor weights (evaluate extremes first)
                        anchor_ws_index = [0, len(weights)-1]
                        anchor_all_rewards = []
                        anchor_bo_grads = []
                        anchor_prev_means = []
                        
                        # Process anchor weights (w=0 and w=1)
                        for idx in anchor_ws_index:
                            latents_i = latents[idx].unsqueeze(0)
                            
                            all_rewards = []
                            # Query multiple samples
                            for k in range(query_size // query_batch_size):
                                prev_latents_noise = all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size]
                                
                                # Generate candidate next latents
                                if sde_type == 'sde':
                                    prev_latents_ik, _, prev_latents_mean, prev_latents_std, prev_latents_dt, prev_latents_noise, = sde_step_with_logprob(
                                        self.scheduler, 
                                        noise_pred[idx].unsqueeze(0).float(), 
                                        t.unsqueeze(0), 
                                        latents_i.float(),
                                        noise_level=noise_level,
                                        sde_type=sde_type,
                                        return_dt=True,
                                        variance_noise=prev_latents_noise,
                                        num_sample_per_step=query_batch_size
                                    )
                                elif sde_type == 'cps':
                                    prev_latents_ik, _, prev_latents_mean, prev_latents_std, prev_latents_noise, = sde_step_with_logprob(
                                        self.scheduler, 
                                        noise_pred[idx].unsqueeze(0).float(), 
                                        t.unsqueeze(0), 
                                        latents_i.float(),
                                        noise_level=noise_level,
                                        sde_type=sde_type,
                                        return_dt=False,
                                        variance_noise=prev_latents_noise,
                                        num_sample_per_step=query_batch_size
                                    )
                                else:
                                    raise NotImplementedError
                                # Store noise if generated
                                if all_prev_latents_noise[k*query_batch_size] is None:
                                    all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size] = prev_latents_noise
                                
                                # Forward pass to next timestep for evaluation
                                latent_model_input = torch.cat([prev_latents_ik] * 2) if do_classifier_free_guidance else prev_latents_ik
                                timestep_next = i+1
                                if timestep_next < len(timesteps):
                                    t_next = timesteps[timestep_next]
                                    timestep_next_expanded = t_next.expand(latent_model_input.shape[0])
                                else:
                                    timestep_next_expanded = torch.zeros(latent_model_input.shape[0], device=timestep.device)
                                
                                prev_noise_pred = self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep_next_expanded,
                                    encoder_hidden_states=prev_prompt_embeds,
                                    pooled_projections=prev_pooled_prompt_embeds,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                                
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = prev_noise_pred.chunk(2)
                                    prev_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                
                                # Get x0 prediction for reward
                                prev_pred_x0 = sde_step_with_logprob(
                                    self.scheduler, 
                                    prev_noise_pred.float(), 
                                    t_next.unsqueeze(0), 
                                    prev_latents_ik.float(),
                                    noise_level=noise_level,
                                    sde_type='original',
                                )
                                prev_pred_x0 = self.vae.decode(
                                    prev_pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                                    return_dict=False,
                                    generator=generator,
                                )[0]
                                
                                # Compute rewards based on anchor weight
                                if weights[idx] == 1:
                                    # Maximize func1
                                    func_scores = self._compute_reward(
                                        prev_pred_x0, opt_func_name[0], pref_function[0],
                                        extra_info, query_batch_size, compress_target, accelerator
                                    )
                                elif weights[idx] == 0:
                                    # Maximize func2
                                    func_scores = self._compute_reward(
                                        prev_pred_x0, opt_func_name[1], pref_function[1],
                                        extra_info, query_batch_size, compress_target, accelerator
                                    )
                                
                                all_rewards.append(func_scores)
                            
                            # Stack noise if list
                            if isinstance(all_prev_latents_noise, list):
                                all_prev_latents_noise = torch.stack(all_prev_latents_noise, dim=0)
                            
                            # Normalize rewards
                            if weights[idx] == 1 or weights[idx] == 0:
                                all_rewards = torch.cat(all_rewards, dim=0)
                                all_rewards_nor = self._normalize_rewards(
                                    all_rewards, reward_norm, weights[idx], opt_func_name, len(timesteps)
                                )
                                all_rewards_nor = all_rewards_nor.reshape(-1, 1, 1, 1)
                                anchor_all_rewards.append(all_rewards_nor)
                            
                            # Compute weighted noise mean for anchor
                            if weights[idx] == 0 and opt_func_name[1] == 'compress' and len(timesteps) == 100:
                                prev_noise_mean = torch.sum(all_rewards_nor[:10] * all_prev_latents_noise[:10], dim=0, keepdim=True)
                            else:
                                prev_noise_mean = torch.sum(all_rewards_nor * all_prev_latents_noise, dim=0, keepdim=True)
                            
                            anchor_bo_grads.append(prev_noise_mean)
                            
                            # Normalize and reconstruct
                            latent_dim = np.prod(latents.shape[1:])
                            if prev_noise_mean.norm() > 0:
                                prev_noise_mean = prev_noise_mean / (prev_noise_mean ** 2).sum(dim=[1,2,3], keepdim=True).sqrt() * (latent_dim ** 0.5)
                            
                            if sde_type == 'sde':
                                prev_sample = prev_latents_mean + prev_latents_std * torch.sqrt(-1*prev_latents_dt) * prev_noise_mean
                            elif sde_type == 'cps':
                                prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            else:
                                raise NotImplementedError

                            prev_latents[idx] = prev_sample
                            anchor_prev_means.append(prev_latents_mean)
                        
                        # Process remaining weights (interpolate between anchors)
                        remaining_ws_indx = np.delete(np.arange(len(weights)), anchor_ws_index)
                        for idx in remaining_ws_indx:
                            latents_i = latents[idx].unsqueeze(0)
                            
                            if sde_type == 'sde':
                                _, _, prev_latents_mean, prev_latents_std, prev_latents_dt, _ = sde_step_with_logprob(
                                    self.scheduler, 
                                    noise_pred[idx].unsqueeze(0).float(), 
                                    t.unsqueeze(0), 
                                    latents_i.float(),
                                    noise_level=noise_level,
                                    sde_type=sde_type,
                                    return_dt=True,
                                    num_sample_per_step=1
                                )
                            elif sde_type == 'cps':
                                _, _, prev_latents_mean, prev_latents_std, _ = sde_step_with_logprob(
                                    self.scheduler, 
                                    noise_pred[idx].unsqueeze(0).float(), 
                                    t.unsqueeze(0), 
                                    latents_i.float(),
                                    noise_level=noise_level,
                                    sde_type=sde_type,
                                    return_dt=False,
                                    num_sample_per_step=1
                                )
                            else:
                                raise NotImplementedError
                            
                            w = weights[idx]
                            
                            # Combine anchor rewards/gradients
                            if cor_flag == False:
                                # No correction
                                w_all_rewards = (w * anchor_all_rewards[1] + (1-w) * anchor_all_rewards[0]).reshape(-1, 1, 1, 1)
                                prev_noise_mean = torch.sum(w_all_rewards * all_prev_latents_noise, dim=0, keepdim=True)
                            else:
                                # With noise correction
                                all_prev_latents_noise_cor_1 = all_prev_latents_noise + (anchor_prev_means[1] - prev_latents_mean)
                                all_prev_latents_noise_cor_1 = all_prev_latents_noise_cor_1 / \
                                    (all_prev_latents_noise_cor_1 ** 2).sum(dim=[1,2,3], keepdim=True).sqrt() * (latent_dim ** 0.5)
                                prev_noise_mean_1 = torch.sum(anchor_all_rewards[1] * all_prev_latents_noise_cor_1, dim=0, keepdim=True)
                                
                                all_prev_latents_noise_cor_2 = all_prev_latents_noise + (anchor_prev_means[0] - prev_latents_mean)
                                all_prev_latents_noise_cor_2 = all_prev_latents_noise_cor_2 / \
                                    (all_prev_latents_noise_cor_2 ** 2).sum(dim=[1,2,3], keepdim=True).sqrt() * (latent_dim ** 0.5)
                                prev_noise_mean_2 = torch.sum(anchor_all_rewards[0] * all_prev_latents_noise_cor_2, dim=0, keepdim=True)
                                
                                prev_noise_mean = w * prev_noise_mean_1 + (1-w) * prev_noise_mean_2
                            
                            # Normalize
                            prev_noise_mean = prev_noise_mean / (prev_noise_mean ** 2).sum(dim=[1,2,3], keepdim=True).sqrt() * (latent_dim ** 0.5)
                            if sde_type == 'sde':
                                prev_sample = prev_latents_mean + prev_latents_std * torch.sqrt(-1*prev_latents_dt) * prev_noise_mean
                            elif sde_type == 'cps':
                                prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            else:
                                raise NotImplementedError
                            prev_latents[idx] = prev_sample
                        
                        prev_latents = torch.cat(prev_latents, dim=0)
                        latents = prev_latents
                        torch.cuda.empty_cache()
                    
                    else:
                        # ============ SIMPLER SOUP (NO ANCHOR) ============
                        prev_latents = []
                        
                        # Initialize noise
                        if opt_func_name[1] == 'compress' and sym_noise_flag:
                            latent_shape = (query_size//2,) + latents.shape[1:]
                            tmp = torch.randn(latent_shape, device=device)
                            all_prev_latents_noise = torch.cat([tmp, -tmp], dim=0)
                        else:
                            all_prev_latents_noise = [None for _ in range(query_size)]
                        
                        # Process each weight
                        for idx, w in enumerate(weights):
                            latents_i = latents[idx].unsqueeze(0)
                            
                            all_rewards_1 = []
                            all_rewards_2 = []
                            
                            # Query samples
                            for k in range(query_size // query_batch_size):
                                prev_latents_noise = all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size]
                                
                                # Generate candidate next latents
                                if sde_type == 'sde':
                                    prev_latents_k, _, prev_latents_mean, prev_latents_std, prev_latents_dt, prev_latents_noise = sde_step_with_logprob(
                                        self.scheduler, 
                                        noise_pred[idx].unsqueeze(0).float(), 
                                        t.unsqueeze(0), 
                                        latents_i.float(),
                                        noise_level=noise_level,
                                        sde_type=sde_type,
                                        return_dt=True,
                                        variance_noise=prev_latents_noise,
                                        num_sample_per_step=query_batch_size
                                    )
                                elif sde_type == 'cps':
                                    prev_latents_k, _, prev_latents_mean, prev_latents_std, prev_latents_noise = sde_step_with_logprob(
                                        self.scheduler, 
                                        noise_pred[idx].unsqueeze(0).float(), 
                                        t.unsqueeze(0), 
                                        latents_i.float(),
                                        noise_level=noise_level,
                                        sde_type=sde_type,
                                        return_dt=False,
                                        variance_noise=prev_latents_noise,
                                        num_sample_per_step=query_batch_size
                                    )
                                else:
                                    raise NotImplementedError
                                
                                if all_prev_latents_noise[k*query_batch_size] is None:
                                    all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size] = prev_latents_noise
                                
                                # Forward pass
                                latent_model_input = torch.cat([prev_latents_k] * 2) if do_classifier_free_guidance else prev_latents_k
                                timestep_next = i + 1
                                if timestep_next < len(timesteps):
                                    t_next = timesteps[timestep_next]
                                    timestep_next_expanded = t_next.expand(latent_model_input.shape[0])
                                else:
                                    timestep_next_expanded = torch.zeros(latent_model_input.shape[0], device=timestep.device)
                                
                                prev_noise_pred = self.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep_next_expanded,
                                    encoder_hidden_states=prev_prompt_embeds,
                                    pooled_projections=prev_pooled_prompt_embeds,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                                
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = prev_noise_pred.chunk(2)
                                    prev_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                # Get x0 prediction for reward
                                prev_pred_x0 = sde_step_with_logprob(
                                    self.scheduler, 
                                    prev_noise_pred.float(), 
                                    t_next.unsqueeze(0), 
                                    prev_latents_k.float(),
                                    noise_level=noise_level,
                                    sde_type='original',
                                )                                
                                prev_pred_x0 = self.vae.decode(
                                    prev_pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                                    return_dict=False,
                                    generator=generator,
                                )[0]
                                
                                # Compute both rewards
                                func1_scores = self._compute_reward(
                                    prev_pred_x0, opt_func_name[0], pref_function[0],
                                    extra_info, query_batch_size, compress_target, accelerator
                                )
                                func2_scores = self._compute_reward(
                                    prev_pred_x0, opt_func_name[1], pref_function[1],
                                    extra_info, query_batch_size, compress_target, accelerator
                                )
                                
                                all_rewards_1.append(func1_scores)
                                all_rewards_2.append(func2_scores)
                            
                            # Stack and normalize
                            if isinstance(all_prev_latents_noise, list):
                                all_prev_latents_noise = torch.stack(all_prev_latents_noise, dim=0)
                            
                            all_rewards_1 = torch.cat(all_rewards_1, dim=0)
                            all_rewards_2 = torch.cat(all_rewards_2, dim=0)
                            
                            # Normalize both rewards
                            all_rewards_1_nor = self._normalize_rewards(all_rewards_1, reward_norm)
                            all_rewards_2_nor = self._normalize_rewards(all_rewards_2, reward_norm)
                            
                            all_rewards_1_nor = all_rewards_1_nor.reshape(-1, 1, 1, 1)
                            all_rewards_2_nor = all_rewards_2_nor.reshape(-1, 1, 1, 1)
                            
                            # Weighted combination
                            prev_noise_mean_1 = torch.sum(all_rewards_1_nor * all_prev_latents_noise, dim=0, keepdim=True)
                            prev_noise_mean_2 = torch.sum(all_rewards_2_nor * all_prev_latents_noise, dim=0, keepdim=True)
                            # if weights[idx] == 0 and opt_func_name[1] == 'compress' and len(timesteps) == 100:
                            #     prev_noise_mean_2 = torch.sum(all_rewards_2_nor[:10] * all_prev_latents_noise[:10], dim=0, keepdim=True)
                            # else:
                            #     prev_noise_mean_2 = torch.sum(all_rewards_2_nor * all_prev_latents_noise, dim=0, keepdim=True)
                            
                            prev_noise_mean = w * prev_noise_mean_1 + (1-w) * prev_noise_mean_2
                            
                            # Normalize
                            latent_dim = np.prod(latents.shape[1:])
                            prev_noise_mean = prev_noise_mean / (prev_noise_mean ** 2).sum(dim=[1,2,3], keepdim=True).sqrt() * (latent_dim ** 0.5)
                            if sde_type == 'sde':
                                prev_sample = prev_latents_mean + prev_latents_std * torch.sqrt(-1*prev_latents_dt) * prev_noise_mean
                            elif sde_type == 'cps':
                                prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            else:
                                raise NotImplementedError
                            prev_latents.append(prev_sample)
                            
                            if i % 10 == 0 or i == len(timesteps)-1:
                                print(f"Step {i}, t={t}, w={w:.2f}")
                                print(f"  Reward1: mean={torch.mean(all_rewards_1):.4f}, max={torch.max(all_rewards_1):.4f}")
                                print(f"  Reward2: mean={torch.mean(all_rewards_2):.4f}, max={torch.max(all_rewards_2):.4f}")
                        
                        prev_latents = torch.cat(prev_latents, dim=0)
                        latents = prev_latents
                        torch.cuda.empty_cache()
                
                else:
                    # ============ STANDARD SAMPLING (NO Guidance) ============
                    latents, _, _, _, _, = sde_step_with_logprob(
                        self.scheduler, 
                        noise_pred.float(), 
                        t.unsqueeze(0), 
                        latents.float(),
                        noise_level=noise_level,
                        sde_type=sde_type,
                        num_sample_per_step=1
                    )
                    
                    # Start divergent sampling
                    if i == divert_start_step - 1:
                        latents = latents.repeat(len(weights), 1, 1, 1)
                        if do_classifier_free_guidance:
                            prompt_embeds = torch.cat([
                                negative_prompt_embeds.repeat(len(weights), 1, 1),
                                log_prompt_embeds.repeat(len(weights), 1, 1),
                            ])
                            pooled_prompt_embeds = torch.cat([
                                negative_pooled_prompt_embeds.repeat(len(weights), 1),
                                log_pooled_prompt_embeds.repeat(len(weights), 1),
                            ])
                
                # Final evaluation
                if i == len(timesteps) - 1:
                    pred_x0 = self.vae.decode(
                        latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    
                    if opt_func_name and opt_func_name[1]:
                        preference_scores = self._compute_reward(
                            pred_x0, opt_func_name[1], pref_function[1],
                            extra_info, pred_x0.shape[0], compress_target, accelerator
                        )
                        
                        if accelerator and accelerator.num_processes > 1:
                            preference_scores = accelerator.gather(preference_scores)
                        
                        preference_score_logs = preference_scores
                        print(f"{opt_func_name[1]} scores:", preference_scores)
                    
                    images = pred_x0
            
            # Progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback_on_step_end is not None:
                    callback_kwargs = {"latents": latents}
                    callback_on_step_end(self, i, t, callback_kwargs)
    
    # 8. Post-processing
    if output_type == 'pil':
        images = self.image_processor.postprocess(images.detach(), output_type=output_type)
    
    return images, preference_score_logs


# Helper methods for reward computation and normalization
def _compute_reward(self, images, func_name, func, extra_info, batch_size, compress_target, accelerator):
    """Compute reward for given function."""
    if func_name == 'ent':
        scores = func(images)
    elif func_name == 'compress':
        scores = func(images)
        scores = torch.tensor(scores, dtype=torch.float).to(accelerator.device if accelerator else images.device)
        if compress_target is not None:
            scores = scores.clamp(max=compress_target)
    elif func_name == 'aes':
        scores = func.score(images)
    elif func_name == 'hps':
        prompts = [extra_info["prompts"]] * batch_size
        scores = func.score(images, prompts)
    elif func_name == 'step_pick':
        # Handle step_pick (not common in SD3, but preserved)
        scores = func(images, extra_info)
    elif func_name == 'clip' or func_name == 'pick':
        prompts = [extra_info["prompts"]] * batch_size
        scores = func(images, prompts)
    else:
        raise NotImplementedError(f"Unknown reward function: {func_name}")
    
    return scores


def _normalize_rewards(self, rewards, norm_method, weight=None, opt_func_name=None, num_timesteps=None):
    """Normalize rewards according to specified method."""
    if norm_method == 'std':
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards)
        if reward_std < 1e-6:
            rewards_nor = torch.zeros_like(rewards, dtype=torch.float32)
            rewards_nor[0] = 1
        else:
            rewards_nor = (rewards - reward_mean) / reward_std
    
    elif norm_method == 'tanh':
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards)
        if reward_std < 1e-6:
            rewards_nor = torch.ones_like(rewards, dtype=torch.float32) / rewards.shape[0]
        else:
            rewards_nor = torch.tanh((rewards - reward_mean) / reward_std)
    
    elif norm_method == 'minmax':
        reward_min = torch.min(rewards)
        reward_max = torch.max(rewards)
        if reward_max - reward_min < 1e-6:
            rewards_nor = torch.ones_like(rewards, dtype=torch.float32) / rewards.shape[0]
        else:
            rewards_nor = (rewards - reward_min) / (reward_max - reward_min)
    
    else:
        raise NotImplementedError(f"Unknown normalization method: {norm_method}")
    
    return rewards_nor

# Add helper methods to the pipeline class
StableDiffusion3Pipeline._compute_reward = _compute_reward
StableDiffusion3Pipeline._normalize_rewards = _normalize_rewards

