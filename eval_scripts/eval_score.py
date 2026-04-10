import argparse
import torch
import numpy as np
import os
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
import torchvision.transforms as transforms
from pathlib import Path

from soup.datasets import d3po_prompt_dataset
from PIL import Image
from matplotlib.colors import CSS4_COLORS
from matplotlib import pyplot as plt
import csv


from soup.utils.rewards import calculate_entropy as ent_fn
from soup.utils.rewards import jpeg_compressibility as compress_fn
from soup.utils.rewards import aesthetic_score as aesthetic_fn
from soup.utils.rewards import PickScore as pick_fn
from soup.utils.rewards import clip_score as clip_fn
from soup.utils.rewards import hps_score as hps_fn
from soup.utils.rewards import ImageReward as ir_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_id', default='SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--prompt', 
        default='an image of a beautiful lake',
    )
    parser.add_argument(
        '--cfg_scale',
        default=7.5,
        type=float,
    )
    parser.add_argument(
        '--output_filename',
        default='spo_sdv1-5_img_release.png',
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    args = parser.parse_args()
    
    pick_scorer = pick_fn(inference_dtype=torch.float32, device=args.device)
    clip_scorer = clip_fn(inference_dtype=torch.float32, device=args.device)
    aes_scorer = aesthetic_fn(inference_dtype=torch.float32, device=args.device)
    hps_scorer = hps_fn(inference_dtype=torch.float32, device=args.device)
    ir_scorer = ir_fn(inference_dtype=torch.float32, device=args.device)

    # Coco dataset
    # dataset = load_dataset("yerevann/coco-karpathy", cache_dir=dataset_cache_dir, split='test')
    # prompts = [entry['sentences'] for entry in dataset][:200]
    # prompts = [item for entry in prompts for item in entry]
    # print(len(prompts), prompts[:5])
    
    # animal dataset
    path = '../assets/prompts/eval_simple_animals.txt'
    # PSD dataset
    #path = '../assets/prompts/hps_v2_all_eval.txt'
    validation_data = d3po_prompt_dataset.from_file(path)  
    prompts = [item for item in validation_data]
    print(len(prompts), prompts[:8])
    #prompts = ['hippopotamus']

    root_dir = '/data/yinghua/projects/Diffusion/SRSoup_Results_ii/'
    method_dir = 'sd_T50_aes_compress_animal_soup_s0e20_ws_query30-10'
    print(method_dir)
    image_dir = root_dir+method_dir +'/3'
    
    weights = np.arange(0, 1.1, 0.2).round(1)
    # weights = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1/3, 1/3, 1/3], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], \
    #     [0.2, 0.2, 0.6], [0.2, 0.6, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2], [0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
    #weights = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    #weights = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    num_epochs = len(weights)
    color_list = list(CSS4_COLORS.keys())[:len(prompts)]
    colors = []
    all_aes_scores = []
    all_hps_scores = []
    all_compress_scores = []
    all_clip_scores = []
    all_pick_scores = []
    all_ir_scores = []
    for e in range(num_epochs):
        aes_scores = []
        compress_scores = []
        ent_scores = []
        hps_scores = []
        clip_scores = []
        pick_scores = []
        ir_scores =  []
        for i, p in enumerate(prompts):
            filename = p[:50]
            if '/' in filename:
                print(p)
            filename = filename.replace('/', '_')
            #image = Image.open(image_dir+'/{}.png'.format(filename, e))
            image = Image.open(image_dir+'/{}_{}.png'.format(filename, weights[e]))
            image = [image]

            #qqqaes_score = aes_u.score(image, p)
            transform = transforms.ToTensor()
            tensor_image = transform(image[0]).to(args.device).unsqueeze(0)
            aes_score = aes_scorer(tensor_image, p)
            aes_scores.append(aes_score.detach().cpu().numpy())
            hps_score = hps_scorer(tensor_image, p)
            hps_scores.append(hps_score.detach().cpu().numpy())
            compress_score = compress_fn()(image)
            compress_scores.append(compress_score)
            image_tensor = torch.tensor(np.array(image[0]), dtype=torch.float).permute(2, 0, 1)/255.
            ent_score = ent_fn(image_tensor.unsqueeze(0))
            ent_scores.append(ent_score.cpu().numpy())
            clip_score = clip_scorer(tensor_image, p)
            clip_scores.append(clip_score.detach().cpu().numpy())
            pick_score = pick_scorer(tensor_image, p)
            pick_scores.append(pick_score.detach().cpu().numpy())
            ir_score = ir_scorer(tensor_image, p)
            ir_scores.append(ir_score.detach().cpu().numpy())
            #print(p, compress_score, aes_score)
        
        print('Mixing weight', weights[e])
        print('AES', len(aes_scores), np.mean(aes_scores))
        print('HPS', len(hps_scores), np.mean(hps_scores))
        print('Compress', len(compress_scores), np.mean(compress_scores))
        print('CLIP', len(clip_scores), np.mean(clip_scores), np.std(clip_scores))
        print('PICK', len(pick_scores), np.mean(pick_scores))
        print('ImageReward', len(ir_scores), np.mean(ir_scores))
        #print('Entropy', len(ent_scores), np.mean(ent_scores))
        # print(aes_scores)
        # print(compress_scores)
        all_aes_scores.append(round(np.mean(aes_scores), 4))
        all_hps_scores.append(round(np.mean(hps_scores), 4))
        all_compress_scores.append(round(np.mean(compress_scores), 4))
        all_clip_scores.append(round(np.mean(clip_scores), 4))
        all_pick_scores.append(round(np.mean(pick_scores), 4))
        all_ir_scores.append(round(np.mean(ir_scores), 4))
    print('AES', all_aes_scores)
    print('HPS', all_hps_scores)
    print('Compress', all_compress_scores)
    print('CLIP', all_clip_scores)
    print('PICK', all_pick_scores)
    print('ImageReward', all_ir_scores)

    csv_path = os.path.join(root_dir, f'{method_dir}_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header with mix_coef values as columns
        writer.writerow(['mix_coef'] + [str(w) for w in weights])
        # Initialize rows for eval_reward_mean and eval_reward_mean2
        writer.writerow(['aes']+all_aes_scores)
        writer.writerow(['compress']+all_compress_scores)
        writer.writerow(['hps']+all_hps_scores)
        writer.writerow(['clip']+all_clip_scores)
        writer.writerow(['pick']+all_pick_scores)
        writer.writerow(['imagereward']+all_ir_scores)
            

if __name__ == '__main__':
    main()
    