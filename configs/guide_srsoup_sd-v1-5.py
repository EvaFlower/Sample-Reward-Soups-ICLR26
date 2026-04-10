from configs.basic_config import basic_config


def aes_hps():
    config = basic_config()
    
    config.mixed_precision = "no"
    config.resolution = 512 
    config.random_crop = False
    config.no_hflip = False
    
    ###### Training ######
    config.sample.num_sample_each_step = 1
    config.sample.guidance_scale = 5.0
    
    config.sample.sample_batch_size = 1
    config.sample.divert_start_step = 0  
    config.sample.divert_end_step = 50
    config.sample.soup_start_step = 0
    config.sample.soup_end_step = 20
    config.sample.num_steps = 50
    config.dataloader_shuffle = False
    config.num_epochs = 1
    config.method = 'fixed_coef'
    config.func_1 = 'aes'
    config.func_2 = 'hps'
    config.sample.eta = 1
    config.reward_norm = 'std'
    config.query_size = 30
    config.query_batch_size = 10  

    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/hps_v2_all_eval.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    #### logging ####
    dataset = 'hpd'
    config.run_name = "sd_T{}_soup_s{}e{}_ws_{}_{}_{}_query{}-{}".format(config.sample.num_steps, config.sample.soup_start_step, \
        config.sample.soup_end_step, config.func_1, config.func_2, dataset, config.query_size, config.query_batch_size)
    config.image_dir = "../SRSoup_Results_ii/"+config.run_name

    return config


def aes_pick():
    config = basic_config()
    config.mixed_precision = 'no'
    
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
    config.func_2 = 'pick'
    config.sample.eta = 1
    config.reward_norm = 'std'
    config.query_size = 30
    config.query_batch_size = 10  #config.query_size//2

    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/hps_v2_all_eval.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    #### logging ####
    dataset = 'hpd'
    config.run_name = "sd_T{}_{}_{}_{}_soup_s{}e{}_ws_query{}-{}".format(config.sample.num_steps, config.func_1, config.func_2, dataset, config.sample.soup_start_step, \
        config.sample.soup_end_step, config.query_size, config.query_batch_size)
    config.image_dir = "../SRSoup_Results_ii/"+config.run_name

    return config


def aes_compress():
    config = basic_config()

    config.mixed_precision = "no"
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
    config.num_epochs = 5
    config.func_1 = 'aes'
    config.func_2 = 'compress'
    config.sample.eta = 1
    config.reward_norm = 'std'
    config.seed = 42
    config.query_size = 30
    config.query_batch_size = 10  #config.query_size//2 

    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/eval_simple_animals.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    #### logging ####
    dataset = 'animal'
    config.run_name = "sd_T{}_{}_{}_{}_soup_s{}e{}_ws_query{}-{}".format(config.sample.num_steps, config.func_1, config.func_2, dataset, config.sample.soup_start_step, \
        config.sample.soup_end_step, config.query_size, config.query_batch_size)
    config.image_dir = "../SRSoup_Results_ii/"+config.run_name

    return config



def aes_hps_pick():
    config = basic_config()
    
    config.mixed_precision = "no"
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
    config.func_1 = 'aes'
    config.func_2 = 'hps'
    config.func_3 = 'pick'
    config.sample.eta = 1
    config.reward_norm = 'std'

    config.dataset_cfg = dict(
        type="D3POPromptDataset",
        meta_json_path='./assets/prompts/hps_v2_all_eval.txt',
        pretrained_tokenzier_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    )
    
    #### logging ####
    dataset = 'hpd'
    config.run_name = "sd_soup_s{}e{}_ws_{}_{}_{}_{}".format(config.sample.soup_start_step, \
        config.sample.soup_end_step, config.func_1, config.func_2, config.func_3, dataset)
    
    config.image_dir = "../SRSoup_Results_ii/"+config.run_name

    return config

def get_config(name):
    return globals()[name]()


