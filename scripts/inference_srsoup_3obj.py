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

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from diffusers import StableDiffusionPipeline, DDIMScheduler
import logging
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from concurrent.futures import ThreadPoolExecutor 

from soup.datasets import build_dataset

from soup.custom_diffusers.pipeline_with_guide_srsoup_3obj import pipeline_with_logprob

from soup.utils.rewards import jpeg_compressibility as compress_fn
from soup.utils.rewards import clip_score
from soup.utils.rewards import PickScore
from soup.utils.rewards import aesthetic_score
from soup.utils.rewards import hps_score 
from soup.utils.profiler import InferenceProfiler, GPUMemoryTracker
from io import StringIO
from contextlib import redirect_stdout

huggingface_cache_dir = '/data/yinghua/projects/Diffusion/SOUP/hugging_face_model'


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
    "./configs/guide_srsoup_sd-v1-5.py:aes_hps_pick", 
    "Training configuration."
)  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    
    device_id = 0 # Change this to the GPU ID you want to use, e.g., 0, 1, etc.
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)

    # timesteps used for inference: [divert_start_step: num_sample_timesteps]
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


    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    print('Inference type: ', inference_dtype)

    # load models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir
    ).to(device)

    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=2,
        disable=False,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(device)  #, dtype=inference_dtype)
    pipeline.text_encoder.to(device)  #, dtype=inference_dtype)
    pipeline.unet.to(device)  #, dtype=inference_dtype)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.sample_batch_size, 1, 1)
    
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    if config.use_lora:
        autocast = contextlib.nullcontext
    else:
        autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext

    # Evaluation before training
    prompt_info = f"Running validation... \n Generating {config.num_validation_images} images with prompt:\n"
    for prompt in config.validation_prompts:
        prompt_info = prompt_info + prompt + '\n'

    logger.info(prompt_info)
    # create pipeline
    pipeline.unet.eval()
    # run inference
    generator = torch.Generator(device=device).manual_seed(config.seed) if config.seed else None

    # Initialize reward functions using factory function
    opt_func_1 = get_reward_function(config.func_1, config, device, inference_dtype)
    opt_func_2 = get_reward_function(config.func_2, config, device, inference_dtype)
    opt_func_3 = get_reward_function(config.func_3, config, device, inference_dtype)
        
    all_preference_scores = []
    rm_scores = []
    pick_scores = []
    hps_scores = []
    aes_scores = []
    compress_scores = []
    i = 0
    weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1/3, 1/3, 1/3], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], \
        [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2], [0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
    #weights = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)
                
    for batch in tqdm(
        data_loader, 
        disable=False,
        desc="Batch",
        position=1,
    ):
        #################### SAMPLING ####################
        pipeline.unet.eval()
                
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

        prompt_embeds1 = pipeline.text_encoder(batch["input_ids"])[0]

        p = batch['prompts'][0]
        extra_info['prompts'] = p
        print(p, device)
        
        filename = p[:50]
        if '/' in filename:
            print(p)
        filename = filename.replace('/', '_')

        #with autocast():
        images, preference_scores = pipeline_with_logprob(
            pipeline,
            prompt_embeds=prompt_embeds1,
            negative_prompt_embeds=sample_neg_prompt_embeds,
            num_inference_steps=config.sample.num_steps,
            guidance_scale=config.sample.guidance_scale,
            eta=config.sample.eta,
            output_type="pil",
            is_noise_images=False,
            pref_function=[opt_func_1, opt_func_2, opt_func_3],  #preference_model_fn,
            opt_func_name=[config.func_1, config.func_2, config.func_3],
            divert_start_step=divert_start_step,
            divert_end_step=divert_end_step,
            soup_start_step=soup_start_step,
            soup_end_step=soup_end_step,
            extra_info=extra_info,
            generator=generator,
            weights=weights,
            reward_norm=config.reward_norm
        )

        for i, im in enumerate(images):
            im.save(config.image_dir+'/{}_{}.png'.format(filename, weights[i]))
        compress_score = compress_fn()(images)
        compress_scores.append(compress_score)
        print('compress', compress_score)
        all_preference_scores.append(preference_scores.cpu().detach().numpy())
        torch.cuda.empty_cache()


    all_preference_scores = np.concatenate(all_preference_scores, axis=0)
    print(config.func_1, np.mean(all_preference_scores), len(all_preference_scores))
    print('compress', np.mean(compress_scores), len(compress_scores))
                
if __name__ == "__main__":
    app.run(main)
