from functools import partial
import copy
import os
import sys
import contextlib
import json
import numpy as np
import tqdm
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, broadcast
from accelerate.logging import get_logger
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from soup.datasets import build_dataset

huggingface_cache_dir = '/data/yinghua/projects/Diffusion/SOUP/hugging_face_model'

from soup.custom_diffusers import (
    ddim_step_with_logprob,
)
from soup.custom_diffusers.pipeline_with_guide_srsoup_sdxl import pipeline_with_score_sdxl

from soup.utils.rewards import jpeg_compressibility as compress_fn
from soup.utils.rewards import clip_score
from soup.utils.rewards import PickScore
from soup.utils.rewards import aesthetic_score
from soup.utils.rewards import hps_score 

def get_reward_function(func_name, config, device, inference_dtype=torch.float32):
    """
    Factory function for reward models.

    Args:
        func_name: Name of the reward function ('aes', 'hps', 'compress', etc.)
        config: Configuration object
        device: Device to use (torch.device or string)

    Returns:
        Initialized reward function

    Raises:
        ValueError: If func_name is not recognized
    """
    if func_name == 'aes':
        return aesthetic_score(device=device)  #aes(dtype=inference_dtype, device=device)
    elif func_name == 'hps':
        return hps_score(device=device)
    elif func_name == 'compress':
        return compress_fn()
    elif func_name == 'clip':
        return clip_score(device=device)
    elif func_name == 'pick':
        return PickScore(device=device)
    else:
        raise ValueError(f"Unknown reward function: {func_name}")

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", 
    "./configs/guide_demon_soup_ws_sdxl.py:aes_pick", 
    "Training configuration."
)  

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    config.sample.num_inner_step = getattr(config.sample, 'num_inner_step', 0)
    
    device_id = 0  # Change this to the GPU ID you want to use, e.g., 0, 1, etc.
    torch.cuda.set_device(device_id)

    # timesteps used for training: [divert_start_step: num_sample_timesteps]
    divert_start_step = config.sample.divert_start_step
    divert_end_step = config.sample.divert_end_step
    soup_start_step = config.sample.soup_start_step
    soup_end_step = config.sample.soup_end_step
    num_train_timesteps = int(config.sample.num_steps)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_config,
        #gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project_name, 
            config=config, 
            init_kwargs={"wandb": {
                "name": config.run_name, 
                "entity": config.wandb_entity_name,
                "mode": "disabled"
                }}
        )
        os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
        with open(os.path.join(config.logdir, config.run_name, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    logger.info(f"\n{config.pretty_text}")

    set_seed(config.seed, device_specific=True)
    
    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float16   # torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load models.
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained.model,
        subfolder="unet",
        cache_dir=huggingface_cache_dir,
    )
    vae_path = (
        config.pretrained.model
        if config.pretrained.vae_model_name_or_path is None
        else config.pretrained.vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if config.pretrained.vae_model_name_or_path is None else None,
        cache_dir=huggingface_cache_dir,
    )
    pipeline.vae = vae
    pipeline.unet = unet
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    if config.use_checkpointing:
        unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=2,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    if config.pretrained.vae_model_name_or_path is None:
        pipeline.vae.to(accelerator.device, dtype=torch.float32)
    else:
        pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        unet.to(accelerator.device, dtype=inference_dtype)
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    prompt_dataset = build_dataset(config.dataset_cfg)
    collate_fn = partial(
        prompt_dataset.sdxl_collate_fn,
        tokenizer=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
    )

    data_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        collate_fn=collate_fn,
        batch_size=config.sample.sample_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    
    # generate negative prompt embeddings
    (
        _, 
        neg_prompt_embed, 
        _, 
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt="",
        device=accelerator.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with `accelerator`.
    unet, data_loader = accelerator.prepare(unet, data_loader)

    # create pipeline
    unet.eval()
    pipeline.unet.eval()
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
    
    opt_func_1 = get_reward_function(config.func_1, config, accelerator.device, inference_dtype)
    opt_func_2 = get_reward_function(config.func_2, config, accelerator.device, inference_dtype)
    
        
    all_preference_scores = []
    rm_scores = []
    pick_scores = []
    hps_scores = []
    aes_scores = []
    compress_scores = []
    i = 0
    weights = np.arange(0, 1.1, 0.2).round(1)
    weights = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #weights = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    #weights = np.array([0.0, 0.5, 1.0])
    
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)
    
    
    for batch in tqdm(
        data_loader, 
        disable=not accelerator.is_local_main_process,
        desc="Batch",
        position=1,
    ):
        #################### SAMPLING ####################
        unet.eval()
        pipeline.unet.eval()
        batch_size = batch['input_ids'].shape[0]
        prompt_ids = batch['input_ids']
        prompt_ids_2 = batch['input_ids_2']
        # encode prompts
        prompt_embeds_list = []
        for i, (text_encoder, text_input_ids) in enumerate(
            zip(
                [pipeline.text_encoder, pipeline.text_encoder_2], 
                [prompt_ids, prompt_ids_2],
            )
        ):
            prompt_embeds = text_encoder(
                text_input_ids, 
                output_hidden_states=True, 
                return_dict=False,
            )
            
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
        
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
        sample_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1)
        
        # prepare extra_info for the preference model
        extra_info = batch['extra_info']
        for k, v in extra_info.items():
            if isinstance(v, torch.Tensor):
                other_dim = [1 for _ in range(v.dim() - 1)]
                extra_info[k] = v.repeat(config.sample.num_sample_each_step, *other_dim)
            elif isinstance(v, list):
                extra_info[k] = v * config.sample.num_sample_each_step
            else:
                raise ValueError(f"Unknown type {type(v)} for extra_info[{k}]")
        p = batch['prompts'][0]
        extra_info['prompts'] = p
        print(p, accelerator.device)
        filename = p[:50]
        if '/' in filename:
            print(p)
        filename = filename.replace('/', '_')
        with autocast():
            images, preference_scores = pipeline_with_score_sdxl(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=sample_negative_pooled_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                pref_function=[opt_func_1, opt_func_2],  #preference_model_fn,
                opt_func_name=[config.func_1, config.func_2],
                divert_start_step=divert_start_step,
                divert_end_step=divert_end_step,
                soup_start_step=soup_start_step,
                soup_end_step=soup_end_step,
                inner_steps=config.sample.num_inner_step,
                extra_info=extra_info,
                accelerator=accelerator,
                weights=weights,
            )
            
        if accelerator.num_processes>1:
            preference_scores = accelerator.gather(preference_scores)

        for i, im in enumerate(images):
            im.save(config.image_dir+'/{}_{}.png'.format(filename, weights[i]))
        compress_score = compress_fn()(images)
        compress_scores.append(compress_score)
        #if torch.isnan(preference_scores):
        if accelerator.is_main_process:
            print(config.func_1, preference_scores, p)
        #print('aes', aes_selector.score(images, p), p)
        print('compress', compress_score)
        all_preference_scores.append(preference_scores.cpu().detach().numpy())
        torch.cuda.empty_cache()

    all_preference_scores = np.concatenate(all_preference_scores, axis=0)
    print(config.func_1, np.mean(all_preference_scores), len(all_preference_scores))
    print('compress', np.mean(compress_scores), len(compress_scores))
                
if __name__ == "__main__":
    app.run(main)
