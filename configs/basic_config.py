import ml_collections

def get_config():
    return basic_config()

def basic_config():
    config = ml_collections.ConfigDict()
    
    ###### General ######
    # random seed for reproducibility.
    config.seed = 42
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = None
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # whether or not to use xFormers to reduce memory usage.
    config.use_xformers = False
    # enable activation checkpointing or not. 
    # this reduces memory usage at the cost of some additional compute.
    config.use_checkpointing = False
    config.mixed_precision = "fp16"
    
    ###### Model Setting ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.use_lora = True
    config.lora_rank = 4
    
    
    ##### dataset #####
    config.dataset_cfg = dict(
        type="PromptDataset",
        meta_json_path='./assets/prompts/4k_training_prompts.json',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    ##### dataloader ####
    config.dataloader_num_workers = 16
    config.dataloader_shuffle = True
    config.dataloader_pin_memory = True
    config.dataloader_drop_last = False

    ###### Training ######
    config.num_epochs = 10
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 20
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    sample.sample_batch_size = 10
    # number of x_{t-1} sampled at each timestep.
    sample.num_sample_each_step = 2

    #### validation ####
    config.validation_prompts = ['A beautiful lake']
    config.num_validation_images = 2
    config.eval_interval = 1
    
    #### logging ####
    # run name for wandb logging and checkpoint saving.
    config.run_name = ""
    config.wandb_project_name = 'guide'
    config.wandb_entity_name = None
    # top-level logging directory for checkpoint saving.
    config.logdir = "work_dirs"
    config.save_interval = 1
    
    return config
