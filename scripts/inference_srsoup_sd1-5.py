from functools import partial
import os
import sys
import contextlib
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

from soup.custom_diffusers.pipeline_with_guide_srsoup_sd1 import pipeline_with_logprob

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
    "./configs/guide_srsoup_sd-v1-5.py:aes_compress", 
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

    # Initialize profiler - only track overall time and memory
    profiler = InferenceProfiler(device=device, enabled=True)
    memory_tracker = GPUMemoryTracker(device=device)
    if profiler is not None:
        profiler.reset()
        memory_tracker.snapshot('start')
        profiler.start('total_process')

    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16
    print('Inference type: ', inference_dtype)

    # load models.
    if profiler is not None:
        memory_tracker.snapshot('Before SD loading')
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir
    ).to(device)
    if profiler is not None:
        memory_tracker.snapshot('After SD loading')

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

    if profiler is not None:
        memory_tracker.snapshot('Before reward model loading')
    # Initialize reward functions using factory function
    opt_func_1 = get_reward_function(config.func_1, config, device, inference_dtype)
    opt_func_2 = get_reward_function(config.func_2, config, device, inference_dtype)
    if profiler is not None:
        memory_tracker.snapshot('After reward model loading')
        
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
    #weights = np.array([0.0])
    
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)
        
    use_parallel_rewards = False
    if use_parallel_rewards:
        max_workers = 2
        reward_executor = ThreadPoolExecutor(max_workers=max_workers)
        print(f"[Parallel Rewards] Using ThreadPoolExecutor with {max_workers} workers")
    else:
        reward_executor = None
    
    num_repeat = 1
    pipeline.unet.eval()

    for n in range(num_repeat):
        n = n
        print('repeat iter: ', n)
        if not os.path.exists(config.image_dir+'/{}'.format(n)):
            os.makedirs(config.image_dir+'/{}'.format(n))
        
        # Start overall profiling
        infer_ctx = profiler.profile(f"total_inference_run-{n}") if profiler else contextlib.nullcontext()
        with infer_ctx:
            print(len(data_loader))
            for batch_idx, batch in enumerate(tqdm(
                data_loader, 
                disable=False,
                desc="Batch",
                position=1,
            )):
                batch_ctx = profiler.profile(f"inference_batch-{batch_idx}") if profiler else contextlib.nullcontext()
                with batch_ctx:
                    #################### SAMPLING ####################
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
                    
                    #print(batch['prompts'])
                    p = batch['prompts'][0]
                    extra_info['prompts'] = p

                    if p == 'lobster':
                        compress_target = -10
                    else:
                        compress_target = -15

                    print(prompt_embeds1.dtype, sample_neg_prompt_embeds.dtype)
                    images, preference_scores = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds1,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pil",
                        is_noise_images=False,
                        pref_function=[opt_func_1, opt_func_2],  #preference_model_fn,
                        opt_func_name=[config.func_1, config.func_2],
                        divert_start_step=divert_start_step,
                        divert_end_step=divert_end_step,
                        soup_start_step=soup_start_step,
                        soup_end_step=soup_end_step,
                        extra_info=extra_info,
                        generator=generator,
                        weights=weights,
                        reward_norm=config.reward_norm,
                        query_size=config.query_size,
                        query_batch_size=config.query_batch_size,
                        reward_executor=reward_executor,
                        profiler=profiler,
                        memory_tracker=memory_tracker,
                        batch_idx=batch_idx,
                        compress_target=compress_target
                    )

                filename = p[:50]
                if '/' in filename:
                    print(p)
                filename = filename.replace('/', '_')
                for i, im in enumerate(images):
                    im.save(config.image_dir+'/{}/{}_{}.png'.format(n, filename, weights[i]))
                
                torch.cuda.empty_cache()

    # End overall profiling
    if profiler is not None:
        profiler.end('total_process')
        memory_tracker.snapshot('end')
    
        # Print and save profiling results
        print("\n" + "="*80)
        print("PROFILING RESULTS")
        print("="*80)
        
        buffer = StringIO()
        # Capture print outputs
        with redirect_stdout(buffer):
            profiler.print_summary()
            memory_tracker.print_breakdown()
        # Write to file
        profile_log_path = os.path.join(config.image_dir, "logs/profiler_output.txt")
        with open(profile_log_path, "w") as f:
            f.write(buffer.getvalue())
        # Also print normally
        print(buffer.getvalue())
        
        # Save profiling results to file
        profiler_path = os.path.join(config.image_dir, "logs/profiling_results.json")
        profiler.save_to_file(profiler_path)
        logger.info(f"Profiling results saved to: {profiler_path}")
        
        # Save memory breakdown
        import json
        memory_breakdown = memory_tracker.get_breakdown()
        memory_snapshots = memory_tracker.snapshots
        memory_path = os.path.join(config.image_dir, "logs/memory_breakdown.json")
        with open(memory_path, 'w') as f:
            json.dump({
                'breakdown': memory_breakdown,
                'snapshots': memory_snapshots
            }, f, indent=2)
        logger.info(f"Memory breakdown saved to: {memory_path}")
                
if __name__ == "__main__":
    app.run(main)
