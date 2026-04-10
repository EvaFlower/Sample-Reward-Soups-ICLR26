from functools import partial
import copy
import os
import sys
import contextlib
import math
import json
import numpy as np
from PIL import Image

import tqdm
import torch
import wandb
from torchvision import transforms
from torchvision.utils import save_image

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
import logging

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from soup.datasets import build_dataset

huggingface_cache_dir = '/data/yinghua/projects/Diffusion/SOUP/hugging_face_model'

from soup.custom_diffusers.pipeline_with_guide_srsoup_sd3 import pipeline_with_logprob

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
    "./configs/guide_demon_soup_ws_sd3.py:aes_pick", 
    "Training configuration."
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    
    # timesteps used for training
    divert_start_step = config.sample.divert_start_step
    divert_end_step = config.sample.divert_end_step
    soup_start_step = config.sample.soup_start_step
    soup_end_step = config.sample.soup_end_step

    # Create output directory and save config
    os.makedirs(os.path.join(config.image_dir, 'logs'), exist_ok=True)
    with open(os.path.join(config.image_dir, "logs/exp_config.py"), "w") as f:
        f.write(config.pretty_text)
    logger.info(f"\n{config.pretty_text}")

    # Set seed
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    # Mixed precision setup
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    print('Inference type: ', inference_dtype)

    # Load SD3 pipeline
    # SD3 uses stabilityai/stable-diffusion-3-medium or stable-diffusion-3-medium-diffusers
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,  # e.g., "stabilityai/stable-diffusion-3-medium-diffusers"
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir
    )
    
    # Load transformer model (replaces UNet in SD3)
    transformer = SD3Transformer2DModel.from_pretrained(
        config.pretrained.model,
        subfolder="transformer",
        cache_dir=huggingface_cache_dir,
    )
    pipeline.transformer = transformer
    
    if config.use_xformers:
        # Note: xformers support may differ for SD3
        pipeline.enable_xformers_memory_efficient_attention()
    
    # Freeze parameters
    pipeline.vae.requires_grad_(False)
    # SD3 has multiple text encoders: text_encoder, text_encoder_2, text_encoder_3
    pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.requires_grad_(False)
    if hasattr(pipeline, 'text_encoder_3') and pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.requires_grad_(False)

    pipeline.transformer.to(device, dtype=inference_dtype)
    pipeline.transformer.requires_grad_(False)
    
    # Disable safety checker
    pipeline.safety_checker = None
    
    # Progress bar config
    pipeline.set_progress_bar_config(
        position=2,
        disable=False,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    
    # SD3 uses FlowMatchEulerDiscreteScheduler, not DDIM
    # The scheduler should already be correct, but ensure it's on the right device
    if hasattr(pipeline.scheduler, 'sigmas'):
        pipeline.scheduler.sigmas = pipeline.scheduler.sigmas.to(device)

    # Move components to device
    pipeline.vae.to(device, dtype=inference_dtype)
    pipeline.text_encoder.to(device, dtype=inference_dtype)
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to(device, dtype=inference_dtype)
    if hasattr(pipeline, 'text_encoder_3') and pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.to(device, dtype=inference_dtype)

    # Enable TF32
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset
    prompt_dataset = build_dataset(config.dataset_cfg)
    collate_fn = partial(
        prompt_dataset.collate_fn,
        tokenizer=pipeline.tokenizer,
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
    
    # Autocast context
    autocast = contextlib.nullcontext 
    # Validation setup
    prompt_info = f"Running validation... \n Generating {config.num_validation_images} images with prompt:\n"
    for prompt in config.validation_prompts:
        prompt_info = prompt_info + prompt + '\n'

    logger.info(prompt_info)
    
    transformer.eval()
    pipeline.transformer.eval()
    
    generator = torch.Generator(device=device).manual_seed(config.seed) if config.seed else None
    
    # Initialize reward models
    opt_func_1 = get_reward_function(config.func_1, config, device, inference_dtype)
    opt_func_2 = get_reward_function(config.func_2, config, device, inference_dtype)
        
    all_preference_scores = []
    compress_scores = []
    
    weights = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)
    
    # Note: NullInversion may need adaptation for SD3's flow matching approach
    # inverter = NullInversion(pipeline, accelerator.device)
    
    num_repeat = 1
    for n in range(num_repeat):
        print('repeat iter: ', n)
        if not os.path.exists(config.image_dir+'/{}'.format(n)):
            os.makedirs(config.image_dir+'/{}'.format(n))
            
        for batch in tqdm(
            data_loader, 
            disable=False,
            desc="Batch",
            position=1,
        ):
            #################### SAMPLING ####################
            transformer.eval()
            pipeline.transformer.eval()
                 
            # Move batch tensors to device
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, torch.Tensor):
                                batch[k][sub_k] = sub_v.to(device)

            extra_info = batch['extra_info']
            for k, v in extra_info.items():
                if isinstance(v, torch.Tensor):
                    other_dim = [1 for _ in range(v.dim() - 1)]
                    extra_info[k] = v.repeat(config.sample.num_sample_each_step, *other_dim)
                elif isinstance(v, list):
                    extra_info[k] = v * config.sample.num_sample_each_step
                else:
                    raise ValueError(f"Unknown type {type(v)} for extra_info[{k}]")

            # SD3 text encoding - get embeddings from all encoders
            with torch.no_grad():
                (
                    prompt_embeds,
                    neg_prompt_embeds,
                    pooled_prompt_embeds,
                    neg_pooled_prompt_embeds
                ) = pipeline.encode_prompt(
                    prompt=batch["prompts"],
                    prompt_2=batch.get("prompts_2", batch["prompts"]),
                    prompt_3=batch.get("prompts_3", batch["prompts"]),
                    device=device,
                    num_images_per_prompt=1,
                )
            
            p = batch['prompts'][0]
            extra_info['prompts'] = p
            print(p, device)
            
            filename = p[:50]
            if '/' in filename:
                print(p)
            filename = filename.replace('/', '_')
            
            with autocast():
                # This function needs significant adaptation for SD3
                # SD3 uses flow matching instead of DDIM/DDPM
                images, preference_scores = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pil",
                    is_noise_images=False,
                    pref_function=[opt_func_1, opt_func_2],
                    opt_func_name=[config.func_1, config.func_2],
                    pref_guide=config.sample.pref_guide,
                    divert_start_step=divert_start_step,
                    divert_end_step=divert_end_step,
                    soup_start_step=soup_start_step,
                    soup_end_step=soup_end_step,
                    extra_info=extra_info,
                    method=config.method,
                    generator=generator,
                    weights=weights,
                    reward_norm=config.reward_norm,
                    query_size=config.query_size,
                    sde_type=config.sde_type,
                    noise_level=config.noise_level
                )

            for i, im in enumerate(images):
                im.save(config.image_dir+'/{}/{}_{}.png'.format(n, filename, weights[i]))
                
            compress_score = compress_fn()(images)
            compress_scores.append(compress_score)
            
            print(config.func_2, preference_scores, p)
            print('compress', compress_score)
            all_preference_scores.append(preference_scores.cpu().detach().numpy())
            torch.cuda.empty_cache()

    all_preference_scores = np.concatenate(all_preference_scores, axis=0)
    print(config.func_1, np.mean(all_preference_scores), len(all_preference_scores))
    print('compress', np.mean(compress_scores), len(compress_scores))
                
if __name__ == "__main__":
    app.run(main)