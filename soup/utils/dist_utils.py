import torch
import torch.distributed as dist

def gather_tensor_with_diff_shape(input_tensor, primary_dim_size_list):
    gathered_tensor_list = [
        input_tensor.new_zeros(
            primary_dim_size, *input_tensor.shape[1:],
        )
        for primary_dim_size in primary_dim_size_list
    ]
    dist.all_gather(gathered_tensor_list, input_tensor)
    gathered_tensor = torch.cat(gathered_tensor_list, dim=0)
    return gathered_tensor


# Helper methods for reward computation and normalization
def compute_reward(images, func_name, func, extra_info, batch_size, device):
    """Compute reward for given function."""
    if func_name == 'compress':
        scores = func(images)
        scores = torch.tensor(scores, dtype=images.dtype).to(device)
    elif func_name == 'aes':
        scores = func(images, '')
    elif func_name == 'hps':
        prompts = [extra_info["prompts"]] * batch_size
        scores = func(images, prompts)
    elif func_name == 'clip' or func_name == 'pick':
        prompts = [extra_info["prompts"]] * batch_size
        scores = func(images, prompts)
    else:
        raise NotImplementedError(f"Unknown reward function: {func_name}")
    
    return scores


# def normalize_reward(all_rewards, reward_norm):
#     if reward_norm == 'std':
#         reward_mean = torch.mean(all_rewards)
#         reward_std = torch.std(all_rewards)
#         if reward_std < 1e-6:
#             all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
#         else:
#             all_rewards_nor = (all_rewards-reward_mean)/reward_std
#     elif reward_norm == 'tanh':
#         tau = 1
#         reward_mean = torch.mean(all_rewards)
#         reward_std = torch.std(all_rewards)
#         if reward_std < 1e-6:
#             all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
#         else:
#             all_rewards_nor = torch.tanh((all_rewards-reward_mean)/reward_std)
#     elif reward_norm == 'minmax':
#         reward_min = torch.min(all_rewards)
#         reward_max = torch.max(all_rewards)
#         if reward_max-reward_min < 1e-6:
#             all_rewards_nor = torch.ones_like(all_rewards, dtype=torch.float32).to(all_rewards.device)/(all_rewards.shape[0])
#         else:
#             all_rewards_nor = (all_rewards-reward_min)/(reward_max-reward_min)
#     else:
#         raise NotImplementedError
    
#     return all_rewards_nor


def normalize_reward(rewards_, norm_method, reward_target=None):
    """Normalize rewards according to specified method."""
    if reward_target is not None:
        mask = rewards_<reward_target
        if mask.sum() == 1:
            rewards_nor_ = torch.where(mask, torch.ones_like(rewards_), torch.zeros_like(rewards_))
            return rewards_nor_
        elif mask.sum() == 0:
            rewards_nor_ = torch.zeros_like(rewards_)
            rewards_nor_[0] = 1
            return rewards_nor_
        else:
            rewards = rewards_[mask]
    else:
        rewards = rewards_
        mask = torch.ones_like(rewards_, dtype=torch.bool)

    if norm_method == 'std':
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards, unbiased=False)+1e-8
        rewards_nor = (rewards_ - reward_mean) / reward_std
            
    elif norm_method == 'tanh':
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards, unbiased=False)+1e-8
        rewards_nor = torch.tanh((rewards_ - reward_mean) / reward_std)
    
    elif norm_method == 'minmax':
        reward_min = torch.min(rewards)
        reward_max = torch.max(rewards)
        rewards_nor = (rewards_ - reward_min) / (reward_max - reward_min+1e-8)
    
    else:
        raise NotImplementedError(f"Unknown normalization method: {norm_method}")
    
    rewards_nor_ = torch.where(mask, rewards_nor, torch.zeros_like(rewards_nor))

    return rewards_nor_

