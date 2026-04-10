from configs.basic_config import basic_config


def aes_hps():
    config = basic_config()
    
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    config.sample.num_inner_step = 1
    
    config.resolution = 512 
    config.random_crop = False
    config.no_hflip = False
    
    ###### Training ######
    config.sample.num_sample_each_step = 1
    config.sample.guidance_scale = 5.0
    
    config.sample.sample_batch_size = 1
    config.sample.divert_start_step = 0  #23,100; 11,50; 49, 200
    config.sample.divert_end_step = 50
    config.sample.soup_start_step = 0
    config.sample.soup_end_step = 20
    config.sample.num_steps = 50
    config.dataloader_shuffle = False
    config.num_epochs = 1
    config.func_1 = 'aes'
    config.func_2 = 'hps'
    config.sample.eta = 1
    config.reward_norm = 'std'

    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/hps_v2_all_eval.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.1,
    )
    
    #### logging ####
    dataset = 'hpd'
    config.run_name = "sdxl_soup_s{}e{}_ws_{}_{}_{}_2".format(config.sample.soup_start_step, \
        config.sample.soup_end_step, config.func_1, config.func_2, dataset)
    config.image_dir = "./results/"+config.run_name

    return config

def aes_pick():
    config = basic_config()
    
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    config.sample.num_inner_step = 1
    
    config.resolution = 512 
    config.random_crop = False
    config.no_hflip = False
    
    ###### Training ######
    config.sample.num_sample_each_step = 1
    config.sample.guidance_scale = 5.0
    
    config.sample.sample_batch_size = 1
    config.sample.divert_start_step = 0  #23,100; 11,50; 49, 200
    config.sample.divert_end_step = 50
    config.sample.soup_start_step = 0
    config.sample.soup_end_step = 0
    config.sample.num_steps = 50
    config.dataloader_shuffle = False
    config.num_epochs = 1
    config.method = 'fixed_coef'
    config.func_1 = 'aes'
    config.func_2 = 'pick'
    config.sample.eta = 1
    
    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/hps_v2_all_eval_2.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    config.compare_func_cfg = dict(
        type="preference_score_compare",
        threshold=0.1,
    )
    
    #### logging ####
    dataset = 'hpd'
    config.run_name = "sdxl_soup_s{}e{}_ws_{}_{}_{}".format(config.sample.soup_start_step, \
        config.sample.soup_end_step, config.func_1, config.func_2, dataset)
    config.image_dir = "./results/"+config.run_name

    return config


def aes_compress():
    config = basic_config()
    
    config.pretrained.model = 'stabilityai/stable-diffusion-xl-base-1.0'
    config.pretrained.vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    
    config.resolution = 512 
    config.random_crop = False
    config.no_hflip = False
    
    ###### Training ######
    config.sample.num_sample_each_step = 1
    config.sample.guidance_scale = 5.0
    
    config.sample.sample_batch_size = 1
    config.sample.divert_start_step = 0  #23,100; 11,50; 49, 200
    config.sample.divert_end_step = 50
    config.sample.soup_start_step = 0
    config.sample.soup_end_step = 20
    config.sample.num_steps = 50
    config.dataloader_shuffle = False
    config.num_epochs = 1
    config.method = 'fixed_coef'
    config.func_1 = 'aes'
    config.func_2 = 'compress'
    config.sample.eta = 1
    config.reward_norm = 'std'
    config.seed = 42
    
    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/eval_simple_animals.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    #### logging ####
    dataset = 'animal'
    config.run_name = "sd_soup_s{}e{}_ws_{}_{}_{}".format(config.sample.soup_start_step, \
        config.sample.soup_end_step, config.func_1, config.func_2, dataset, config.seed)
    
    config.image_dir = "./results/"+config.run_name

    return config


def get_config(name):
    return globals()[name]()


