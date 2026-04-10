import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from .builder import DATASETS

from soup.utils import huggingface_cache_dir
from datasets import load_dataset


def _load_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path):
    prompts = _load_lines(path)
    return prompts


@DATASETS.register_module()
class PromptDataset(Dataset):
    def __init__(self, meta_json_path, pretrained_tokenzier_path, caption_key='caption'):
        if meta_json_path == 'pickapic_valid':
            self.meta = load_dataset("yuvalkirstain/pickapic_v1_no_images", cache_dir='hugging_face_dataset', \
                split='validation_unique')  
        else:
            with open(meta_json_path, 'r') as f:
                self.meta = json.load(f)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenzier_path,
            cache_dir=huggingface_cache_dir,
        )
        self.caption_key = caption_key

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        info = self.meta[idx]
        prompt = info[self.caption_key]
        # input_ids
        # attention_mask
        preference_model_input_ids = self.clip_tokenizer(
            prompt,
            max_length=self.clip_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        sample = {
            "prompt": prompt,
            "preference_model_input_ids": preference_model_input_ids,
        }
        return sample
    
    @staticmethod
    def collate_fn(examples, tokenizer):
        prompts = [item['prompt'] for item in examples]
        preference_model_input_ids = [item['preference_model_input_ids'] for item in examples]
        preference_model_input_ids = torch.cat(preference_model_input_ids, dim=0)
        extra_info = {
            'input_ids': preference_model_input_ids,
        }
        input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        
        return dict(
            prompts=prompts,
            input_ids=input_ids,
            extra_info=extra_info,
        )

    @staticmethod
    def sdxl_collate_fn(examples, tokenizer, tokenizer_2):
        prompts = [item['prompt'] for item in examples]
        preference_model_input_ids = [item['preference_model_input_ids'] for item in examples]
        preference_model_input_ids = torch.cat(preference_model_input_ids, dim=0)
        extra_info = {
            'input_ids': preference_model_input_ids,
        }
        input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        input_ids_2 = tokenizer_2(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
        
        return dict(
            prompts=prompts,
            input_ids=input_ids,
            input_ids_2=input_ids_2,
            extra_info=extra_info,
        )
