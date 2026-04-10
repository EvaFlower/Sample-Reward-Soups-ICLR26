from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import numpy as np

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    rescale_noise_cfg,
    retrieve_timesteps,
    is_torch_xla_available,
)
from .ddim_seperate_func import ddim_step_fetch_x0, ddim_step_fetch_x_t_1
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
from soup.utils.dist_utils import compute_reward, normalize_reward


@torch.no_grad()
def pipeline_with_score_sdxl(
    self: StableDiffusionXLPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    callback=None,
    callback_steps=None,
    
    divert_start_step=0,
    divert_end_step=50,
    soup_start_step=0,
    soup_end_step=0,
    extra_info=None,
    pref_function=None,
    opt_func_name=None,
    inner_steps=3,
    accelerator=None,
    weights=None,
    reward_norm='std',
    cor_flag=True
):
    # 0. Default height and width to unet
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._denoising_end = denoising_end
    self._interrupt = False


    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )
    log_prompt_embeds = prompt_embeds
    log_add_text_embeds = pooled_prompt_embeds

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
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

        
    query_size = 30
    query_batch_size = 6
    assert query_size % query_batch_size == 0
    
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids
    add_time_ids = add_time_ids.to(device)
    log_add_time_ids = add_time_ids
    negative_add_time_ids = negative_add_time_ids.to(device)

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        prev_prompt_embeds = torch.cat([
            negative_prompt_embeds.repeat(query_batch_size, 1, 1),
            log_prompt_embeds.repeat(query_batch_size, 1, 1),
        ])
        prev_add_text_embeds = torch.cat([
            negative_pooled_prompt_embeds.repeat(query_batch_size, 1),
            log_add_text_embeds.repeat(query_batch_size, 1),
        ], dim=0)
        prev_add_time_ids = torch.cat([
            negative_add_time_ids,
            log_add_time_ids,
        ], dim=0).repeat(query_batch_size*batch_size*num_images_per_prompt, 1)
    else:
        prev_prompt_embeds = log_prompt_embeds.repeat(query_batch_size, 1, 1)
        prev_add_text_embeds = add_text_embeds.repeat(query_batch_size, 1)
        prev_add_time_ids = add_time_ids.repeat(query_batch_size, 1)
        

    # prompt_embeds = prompt_embeds.to(device)
    # add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    # negative_add_time_ids = negative_add_time_ids.to(device)

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    all_latents = [latents]
    
    # ignored 8.1 Apply denoising_end
    
    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    
    denoise_idx = None
    
    preference_score_logs = []

    
    if divert_start_step == 0:
        latents = latents.repeat(len(weights), 1, 1, 1)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([
                negative_prompt_embeds.repeat(len(weights), 1, 1),
                log_prompt_embeds.repeat(len(weights), 1, 1),
            ])
            add_text_embeds = torch.cat([
                negative_pooled_prompt_embeds.repeat(len(weights), 1),
                log_add_text_embeds.repeat(len(weights), 1),
            ], dim=0)
            add_time_ids = torch.cat([
                negative_add_time_ids,
                log_add_time_ids,
            ], dim=0).repeat(len(weights)*batch_size*num_images_per_prompt, 1)
        else:
            prompt_embeds = log_prompt_embeds.repeat(len(weights), 1, 1)
            add_text_embeds = log_add_text_embeds.repeat(len(weights), 1)
            add_time_ids = log_add_time_ids.repeat(len(weights)*batch_size*num_images_per_prompt, 1)
    
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            for j in range(inner_steps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                
                if divert_end_step > i >= divert_start_step:
                    
                    if  soup_start_step <= i < soup_end_step:    # i < ws_start_step
                        prev_latents = [[] for i in range(len(weights))]      
                        all_prev_latents_noise = [None for i in range(query_size)]
                        # ## Latin Hypercube Sampling
                        # from scipy.stats import qmc      
                        # _, c, h, w = latents.shape      
                        # sampler = qmc.LatinHypercube(d=c*h*w)
                        # sample = sampler.random(n=query_size)  # 采样 [0,1] 区间
                        # all_prev_latents_noise = torch.tensor(qmc.scale(sample, -1, 1), dtype=latents.dtype, device=latents.device)  # 映射到 [-1,1]
                        # all_prev_latents_noise = all_prev_latents_noise.view(query_size, c, h, w)
                        anchor_ws_index = [0, len(weights)-1]
                        anchor_all_rewards = []
                        anchor_bo_grads = []
                        anchor_prev_means = []
                        for idx in anchor_ws_index:     
                            latents_i = latents[idx].unsqueeze(0)
                            pred_dict = ddim_step_fetch_x0(
                                self.scheduler,
                                noise_pred[idx].unsqueeze(0),
                                t,
                                latents_i,
                            )
                            all_rewards = []
                            all_rewards_1 = []
                            all_rewards_2 = []
                            
                            for k in range(query_size//query_batch_size):
                                prev_latents_noise = all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size]
                                prev_latents_ik, prev_latents_mean, prev_latents_std, prev_latents_noise = ddim_step_fetch_x_t_1(
                                    self.scheduler,
                                    dtype=latents.dtype,
                                    num_sample_per_step=query_batch_size,
                                    timestep=t,
                                    return_prev_mean=True,
                                    variance_noise=prev_latents_noise,
                                    **extra_step_kwargs,
                                    **pred_dict,  # noise is same for different weightings, but latents_k are different.
                                )
                                if all_prev_latents_noise[k*query_batch_size] is None:
                                    all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size] = prev_latents_noise
                                else:
                                    pass
                                latent_model_input = torch.cat([prev_latents_ik] * 2) if self.do_classifier_free_guidance else prev_latents_ik
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, pred_dict['prev_timestep'])
                                # predict the noise residual
                                prev_added_cond_kwargs = {"text_embeds": prev_add_text_embeds, "time_ids": prev_add_time_ids}
                                torch.cuda.empty_cache()
                                prev_noise_pred = self.unet(
                                    latent_model_input,
                                    pred_dict['prev_timestep'],
                                    encoder_hidden_states=prev_prompt_embeds,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=prev_added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                                # perform guidance
                                if self.do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = prev_noise_pred.chunk(2)
                                    prev_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                    prev_noise_pred = rescale_noise_cfg(prev_noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                                prev_pred_dict = ddim_step_fetch_x0(
                                    self.scheduler,
                                    prev_noise_pred,
                                    pred_dict['prev_timestep'],
                                    prev_latents_ik,
                                )
                                # num_sample_per_step+1, c, h, w
                                prev_pred_x0 = prev_pred_dict['pred_original_sample']
                                prev_pred_x0 = self.vae.decode(
                                    prev_pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                                    return_dict=False,
                                    generator=generator,
                                )[0]
                                prev_pred_x0 = prev_pred_x0.float()
                                if weights[idx] == 1:
                                    func1_scores = compute_reward(
                                        prev_pred_x0, opt_func_name[0], pref_function[0],
                                        extra_info, query_batch_size, device
                                    )
                                    all_rewards.append(func1_scores)
                                elif weights[idx] == 0:
                                    func2_scores = compute_reward(
                                        prev_pred_x0, opt_func_name[1], pref_function[1],
                                        extra_info, query_batch_size, device
                                    )
                                    all_rewards.append(func2_scores)
                                else:  #for w!=0 or w!=1
                                    pass                     
                                
                            if isinstance(all_prev_latents_noise, list):
                                all_prev_latents_noise = torch.stack(all_prev_latents_noise, dim=0)
                            if weights[idx] == 1 or weights[idx] == 0:
                                all_rewards = torch.cat(all_rewards, dim=0)
                                if reward_norm == 'std':
                                    reward_mean = torch.mean(all_rewards)
                                    reward_std = torch.std(all_rewards)
                                    if reward_std < 1e-6:
                                        all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
                                    else:
                                        all_rewards_nor = (all_rewards-reward_mean)/reward_std
                                elif reward_norm == 'tanh':
                                    tau = 1
                                    reward_mean = torch.mean(all_rewards)
                                    reward_std = torch.std(all_rewards)
                                    if reward_std < 1e-6:
                                        all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
                                    else:
                                        all_rewards_nor = torch.tanh((all_rewards-reward_mean)/reward_std)
                                elif reward_norm == 'minmax':
                                    reward_min = torch.min(all_rewards)
                                    reward_max = torch.max(all_rewards)
                                    if reward_max-reward_min < 1e-6:
                                        all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
                                    else:
                                        all_rewards_nor = (all_rewards-reward_min)/(reward_max-reward_min)
                                else:
                                    raise NotImplementedError
                                all_rewards_nor = all_rewards_nor.reshape(-1, 1, 1, 1)
                                anchor_all_rewards.append(all_rewards_nor)
                                if i % 10 == 0 or i == len(timesteps)-1:
                                    print(torch.mean(all_rewards), torch.max(all_rewards), t)
                            else:   #for w!=0 or w!=1
                                pass
                            prev_noise_mean = torch.sum(all_rewards_nor * all_prev_latents_noise, dim=0, keepdim=True)
                            anchor_bo_grads.append(prev_noise_mean)  # Do not normalize respective grads.
                            latent_dim = np.prod(latents.shape[1:])
                            prev_noise_mean = prev_noise_mean/(prev_noise_mean ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()*(latent_dim**0.5)
                            prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            prev_latents_i = prev_sample
                            prev_latents[idx] = prev_latents_i
                            anchor_prev_means.append(prev_latents_mean)
                            
                        remaining_ws_indx = np.delete(np.arange(len(weights)), anchor_ws_index)
                        for idx in remaining_ws_indx:
                            latents_i = latents[idx].unsqueeze(0)
                            pred_dict = ddim_step_fetch_x0(
                                self.scheduler,
                                noise_pred[idx].unsqueeze(0),
                                t,
                                latents_i,
                            )
                            _, prev_latents_mean, prev_latents_std, _ = ddim_step_fetch_x_t_1(
                                    self.scheduler,
                                    dtype=latents.dtype,
                                    num_sample_per_step=query_batch_size,
                                    timestep=t,
                                    return_prev_mean=True,
                                    variance_noise=prev_latents_noise,
                                    **extra_step_kwargs,
                                    **pred_dict,  # noise is same for different weightings, but latents_k are different.
                                )
                            
                            w = weights[idx]
                            if cor_flag == False:
                                w_all_rewards = (w*anchor_all_rewards[1]+(1-w)*anchor_all_rewards[0]).reshape(-1, 1, 1, 1)
                                prev_noise_mean = torch.sum(w_all_rewards * all_prev_latents_noise, dim=0, keepdim=True)
                            else:
                                all_prev_latents_noise_cor_1 = (all_prev_latents_noise+(anchor_prev_means[1]-prev_latents_mean))
                                all_prev_latents_noise_cor_1 = all_prev_latents_noise_cor_1/(all_prev_latents_noise_cor_1 ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()*(latent_dim**0.5)
                                prev_noise_mean_1 = torch.sum(anchor_all_rewards[1]*all_prev_latents_noise_cor_1, dim=0, keepdim=True)
                                all_prev_latents_noise_cor_2 = (all_prev_latents_noise+(anchor_prev_means[0]-prev_latents_mean))
                                all_prev_latents_noise_cor_2 = all_prev_latents_noise_cor_2/(all_prev_latents_noise_cor_2 ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()*(latent_dim**0.5)
                                prev_noise_mean_2 = torch.sum(anchor_all_rewards[0]*all_prev_latents_noise_cor_2, dim=0, keepdim=True)
                                prev_noise_mean = w*prev_noise_mean_1+(1-w)*prev_noise_mean_2
                            latent_dim = np.prod(latents.shape[1:])
                            prev_noise_mean = prev_noise_mean/(prev_noise_mean ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()*(latent_dim**0.5)
                            prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            prev_latents_i = prev_sample
                            prev_latents[idx] = prev_latents_i
                                
                        prev_latents = torch.cat(prev_latents, dim=0)
                        latents = prev_latents.to(latents.dtype)
                        torch.cuda.empty_cache()
                    else:
                        prev_latents = []      
                        all_prev_latents_noise = [None for i in range(query_size)]
                        for idx, w in enumerate(weights):     
                            latents_i = latents[idx].unsqueeze(0)
                            pred_dict = ddim_step_fetch_x0(
                                self.scheduler,
                                noise_pred[idx].unsqueeze(0),
                                t,
                                latents_i,
                            )
                            
                            all_rewards_1 = []
                            all_rewards_2 = []
                            for k in range(query_size//query_batch_size):
                                prev_latents_noise = all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size]
                                prev_latents_k, prev_latents_mean, prev_latents_std, prev_latents_noise = ddim_step_fetch_x_t_1(
                                    self.scheduler,
                                    dtype=latents.dtype,
                                    num_sample_per_step=query_batch_size,
                                    timestep=t,
                                    return_prev_mean=True,
                                    variance_noise=prev_latents_noise,
                                    **extra_step_kwargs,
                                    **pred_dict,  # noise is same for different weightings, but latents_k are different.
                                )
                                if all_prev_latents_noise[k*query_batch_size] is None:
                                    all_prev_latents_noise[k*query_batch_size:(k+1)*query_batch_size] = prev_latents_noise
                                else:
                                    pass
                                latent_model_input = torch.cat([prev_latents_k] * 2) if self.do_classifier_free_guidance else prev_latents_k
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, pred_dict['prev_timestep'])
                                # predict the noise residual
                                prev_added_cond_kwargs = {"text_embeds": prev_add_text_embeds, "time_ids": prev_add_time_ids}
                                torch.cuda.empty_cache()
                                prev_noise_pred = self.unet(
                                    latent_model_input,
                                    pred_dict['prev_timestep'],
                                    encoder_hidden_states=prev_prompt_embeds,
                                    timestep_cond=timestep_cond,
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=prev_added_cond_kwargs,
                                    return_dict=False,
                                )[0]
                                # perform guidance
                                if self.do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = prev_noise_pred.chunk(2)
                                    prev_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                                    prev_noise_pred = rescale_noise_cfg(prev_noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
                                prev_pred_dict = ddim_step_fetch_x0(
                                    self.scheduler,
                                    prev_noise_pred,
                                    pred_dict['prev_timestep'],
                                    prev_latents_k,
                                )
                                # num_sample_per_step+1, c, h, w
                                prev_pred_x0 = prev_pred_dict['pred_original_sample']
                                prev_pred_x0 = self.vae.decode(
                                    prev_pred_x0.to(self.vae.dtype) / self.vae.config.scaling_factor,
                                    return_dict=False,
                                    generator=generator,
                                )[0]
                                prev_pred_x0 = prev_pred_x0.float()
                                func1_scores = compute_reward(
                                    prev_pred_x0, opt_func_name[0], pref_function[0],
                                    extra_info, query_batch_size, device
                                )
                                all_rewards_1.append(func1_scores)
                                func2_scores = compute_reward(
                                    prev_pred_x0, opt_func_name[1], pref_function[1],
                                    extra_info, query_batch_size, device
                                )
                                all_rewards_2.append(func2_scores)
                            if isinstance(all_prev_latents_noise, list):
                                all_prev_latents_noise = torch.stack(all_prev_latents_noise, dim=0)
                            all_rewards_1 = torch.cat(all_rewards_1, dim=0)
                            all_rewards_2 = torch.cat(all_rewards_2, dim=0)

                            if reward_norm == 'std':
                                reward1_mean = torch.mean(all_rewards_1)
                                reward1_std = torch.std(all_rewards_1)
                                reward2_mean = torch.mean(all_rewards_2)
                                reward2_std = torch.std(all_rewards_2)
                                if reward1_std < 1e-6:
                                    all_rewards_1_nor = torch.ones_like(all_rewards_1, dtype=torch.float32).to(all_rewards_1.device)/(all_rewards_1.shape[0])
                                else:
                                    all_rewards_1_nor = (all_rewards_1-reward1_mean)/reward1_std
                                if reward2_std < 1e-6:
                                    all_rewards_2_nor = torch.ones_like(all_rewards_2, dtype=torch.float32).to(all_rewards_2.device)/(all_rewards_2.shape[0])
                                else:
                                    all_rewards_2_nor = (all_rewards_2-reward2_mean)/reward2_std
                            elif reward_norm == 'tanh':
                                tau = 1
                                reward1_mean = torch.mean(all_rewards_1)
                                reward1_std = torch.std(all_rewards_1)
                                reward2_mean = torch.mean(all_rewards_2)
                                reward2_std = torch.std(all_rewards_2)
                                if reward1_std < 1e-6:
                                    all_rewards_1_nor = torch.ones_like(all_rewards_1, dtype=torch.float32).to(all_rewards_1.device)/(all_rewards_1.shape[0])
                                else:
                                    all_rewards_1_nor = torch.tanh((all_rewards_1-reward1_mean)/reward1_std)
                                if reward2_std < 1e-6:
                                    all_rewards_2_nor = torch.ones_like(all_rewards_2, dtype=torch.float32).to(all_rewards_2.device)/(all_rewards_2.shape[0])
                                else:
                                    all_rewards_2_nor = torch.tanh((all_rewards_2-reward2_mean)/reward2_std)
                            elif reward_norm == 'minmax':
                                reward1_min = torch.min(all_rewards_1)
                                reward1_max = torch.max(all_rewards_1)
                                reward2_min = torch.min(all_rewards_2)
                                reward2_max = torch.max(all_rewards_2)
                                if reward1_max-reward1_min < 1e-6:
                                    all_rewards_1_nor = torch.ones_like(all_rewards_1, dtype=torch.float32).to(all_rewards_1.device)/(all_rewards_1.shape[0])
                                else:
                                    all_rewards_1_nor = (all_rewards_1-reward1_min)/(reward1_max-reward1_min)
                                if reward2_max-reward2_min < 1e-6:
                                    all_rewards_2_nor = torch.ones_like(all_rewards_2, dtype=torch.float32).to(all_rewards_2.device)/(all_rewards_2.shape[0])
                                else:
                                    all_rewards_2_nor = (all_rewards_2-reward2_min)/(reward2_max-reward2_min)
                            else:
                                raise NotImplementedError


                            w_all_rewards = (w*all_rewards_1_nor+(1-w)*all_rewards_2_nor).reshape(-1, 1, 1, 1)
                            prev_noise_mean = torch.sum(w_all_rewards * all_prev_latents_noise, dim=0, keepdim=True)
                            latent_dim = np.prod(latents.shape[1:])
                            prev_noise_mean = prev_noise_mean/(prev_noise_mean ** 2).sum(axis=[1, 2, 3], keepdim=True).sqrt()*(latent_dim**0.5)
                            prev_sample = prev_latents_mean + prev_latents_std * prev_noise_mean
                            prev_latents_k = prev_sample
                            prev_latents.append(prev_latents_k)
                            if i % 10 == 0 or i == len(timesteps)-1:
                                print(torch.mean(all_rewards_1), torch.max(all_rewards_1), t, w)
                                print(torch.mean(all_rewards_2), torch.max(all_rewards_2), t, w)
                        prev_latents = torch.cat(prev_latents, dim=0)
                        latents = prev_latents.to(latents.dtype)
                        torch.cuda.empty_cache()
                else:
                    pred_dict = ddim_step_fetch_x0(
                        self.scheduler,
                        noise_pred, 
                        t, 
                        latents, 
                    )
                    latents = ddim_step_fetch_x_t_1(
                        self.scheduler,
                        dtype=latents.dtype,
                        num_sample_per_step=1,
                        timestep=t,
                        **extra_step_kwargs,
                        **pred_dict,
                    )
                    if i == divert_start_step-1:
                        latents = latents.repeat(len(weights), 1, 1, 1)
                        if self.do_classifier_free_guidance:
                            prompt_embeds = torch.cat([
                                negative_prompt_embeds.repeat(len(weights), 1, 1),
                                log_prompt_embeds.repeat(len(weights), 1, 1),
                            ])        
                            add_text_embeds = torch.cat([
                                negative_pooled_prompt_embeds.repeat(len(weights), 1),
                                log_add_text_embeds.repeat(len(weights), 1),
                            ], dim=0)
                            add_time_ids = torch.cat([
                                negative_add_time_ids,
                                log_add_time_ids,
                            ], dim=0).repeat(len(weights)*batch_size*num_images_per_prompt, 1)
                        else:
                            prompt_embeds = log_prompt_embeds.repeat(len(weights), 1, 1)
                            add_text_embeds = log_add_text_embeds.repeat(len(weights), 1)
                            add_time_ids = log_add_time_ids.repeat(len(weights), 1)
                              
                            
                if i == len(timesteps) - 1:
                    if latents.shape[0]>query_batch_size:
                        n_bs = (latents.shape[0] + query_batch_size - 1) // query_batch_size
                        pred_x0 = []
                        for ii in range(n_bs):
                            # num_sample_per_step*b, c, h, w
                            tmp = self.vae.decode(
                                latents[ii*query_batch_size:(ii+1)*query_batch_size].to(self.vae.dtype) / self.vae.config.scaling_factor,
                                return_dict=False,
                                generator=generator,
                            )[0]
                            pred_x0.append(tmp)
                        pred_x0 = torch.cat(pred_x0, dim=0)
                    else:
                        # num_sample_per_step*b, c, h, w
                        pred_x0 = self.vae.decode(
                            latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                            return_dict=False,
                            generator=generator,
                        )[0]
                    pred_x0 = pred_x0.float()
                    preference_scores = compute_reward(
                        prev_pred_x0, opt_func_name[1], pref_function[1],
                        extra_info, query_batch_size, device
                    )
                    images = pred_x0
                    preference_score_logs = preference_scores

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

    if not output_type == "latent":
        if self.safety_checker is not None:
            images, has_nsfw_concept = self.safety_checker(images, device, prompt_embeds.dtype)
        else:
            has_nsfw_concept = None
        
    if has_nsfw_concept is None:
        do_denormalize = [True] * images.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
                
    if output_type == 'pil':
        images = self.image_processor.postprocess(images.detach(), output_type=output_type, do_denormalize=do_denormalize)                
    
    return images, preference_score_logs
