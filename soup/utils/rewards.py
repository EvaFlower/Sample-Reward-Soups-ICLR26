from PIL import Image
import io
import numpy as np
import torch
import torch.nn.functional as F
#from diff_jpeg import diff_jpeg_coding


def kde_batch(input_tensor, bandwidth=0.1, bins=100, range_min=0.0, range_max=1.0):
    # Flatten spatial dimensions (256, 256) for each channel in every batch
    batch_size, n_channel, height, width = input_tensor.shape
    input_tensor = input_tensor.view(batch_size, n_channel, -1)  # Shape: (batch_size, n_channel, 256*256)
    
    # Normalize input tensor between [0, 1]
    input_tensor = (input_tensor + 1) / 2
    input_tensor = input_tensor.clamp(0, 1)
    
    # Create evenly spaced bins (as centers for KDE evaluation)
    bin_centers = torch.linspace(range_min, range_max, bins).to(input_tensor.device)
    
    all_densities = []
    
    # Perform KDE for each batch and each channel independently
    for batch_idx in range(batch_size):
        for channel_idx in range(n_channel):
            data = input_tensor[batch_idx, channel_idx]  # Shape: (256*256)
            
            # Compute pairwise differences between data points and bin centers
            diff = data.unsqueeze(1) - bin_centers.unsqueeze(0)
            
            # Apply Gaussian kernel with specified bandwidth
            kernel_vals = torch.exp(- (diff ** 2) / (2 * bandwidth ** 2))
            
            # Sum kernel values to get the density estimate for each bin
            density_estimate = kernel_vals.sum(dim=0)
            
            # Normalize the density estimate
            density_estimate /= (density_estimate.sum() * (range_max - range_min) / bins)
            
            all_densities.append(density_estimate)
    
    # Stack densities and reshape for batch/channel structure
    all_densities = torch.stack(all_densities).view(batch_size, n_channel, bins)
    
    return all_densities, bin_centers


def differentiable_histogram(input, bins=100, min=None, max=None):
    """ Compute a differentiable histogram of a tensor. 
        The function returns a tensor of shape (batch_size, n_channels, bins) where each value represents the count of values in the input tensor that fall into the corresponding bin.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, n_channels, ...)
            bins (int): Number of bins in the histogram.
            min (Optional[float]): Minimum value of the histogram (inclusive). If None, the minimum value of the input tensor is used.
            max (Optional[float]): Maximum value of the histogram (inclusive). If None, the maximum value of the input tensor is used.
        Returns:
            hist (torch.Tensor): Histogram tensor of shape (batch_size, n_channels, bins)
    """
    # Ensure the input tensor has at least 2 dimensions
    assert input.ndim >= 2
    input = input.view(input.shape[0], input.shape[1], -1)
    batch_size, n_channels, n_values = input.shape

    # Compute the minimum and maximum values of the input tensor
    if min is None:
        min = input.min().item()
    if max is None:
        max = input.max().item()
    
    # Initialize the histogram tensor
    hist = torch.zeros(batch_size, n_channels, bins).to(input.device)

    # Create a table of bin edges
    delta = (max - min) / bins
    BIN_Table = torch.arange(start=0, end=bins+1, step=1) * delta

    # Iterate over each bin
    for dim in range(1, bins-1):
        h_curr = BIN_Table[dim].item()
        h_last = BIN_Table[dim - 1].item()
        h_next = BIN_Table[dim + 1].item()

        # Create masks for values falling into the current bin
        mask_last = ((h_last <= input) & (input < h_curr)).float()
        mask_next = ((h_curr <= input) & (input <= h_next)).float()

        # Accumulate histogram values for the current bin
        hist[:, :, dim] += torch.sum(((input - h_last) * mask_last).view(batch_size, n_channels, -1), dim=-1)
        hist[:, :, dim] += torch.sum(((h_next - input) * mask_next).view(batch_size, n_channels, -1), dim=-1)
    
    # Handle the first bin
    mask = (input < BIN_Table[1]).float()
    hist[:, :, 0] += torch.sum(((BIN_Table[1] - input) * mask).view(batch_size, n_channels, -1), dim=-1)

    # Handle the last bin
    mask = (input >= BIN_Table[bins-1]).float()
    hist[:, :, bins-1] += torch.sum(((input - BIN_Table[bins-1]) * mask).view(batch_size, n_channels, -1), dim=-1)

    # Divide by the bin width
    hist = hist / delta

    # Normalize the histogram
    hist = hist / hist.sum(dim=-1, keepdim=True) * n_values
    
    return hist

def calculate_entropy(image_tensor):
    # Flatten the image tensor to 1D, image range [0, 1]
    # if image_tensor.min()<0 and image_tensor.max()>=1:
    #     image_tensor = (image_tensor+1)/2
    #     image_tensor = image_tensor.clamp(0, 1) 
    image_tensor = (image_tensor+1)/2
    image_tensor = image_tensor.clamp(0, 1) 
    hist = differentiable_histogram(image_tensor, bins=256, min=0, max=1)
    hist_prob = hist / hist.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(hist_prob * torch.log2(hist_prob + 1e-6), dim=-1).mean(dim=-1)
    neg_entropy = -entropy
    
    return neg_entropy


def diff_jpeg_compress(image_tensor, jpeg_quality=25):
    image_tensor = (image_tensor+1)/2
    image_tensor = image_tensor.clamp(0, 1)*255.
    jpeg_quality = torch.tensor([jpeg_quality]).to(image_tensor.device)
    # Perform differentiable JPEG coding
    image_coded, y_encoded, cb_encoded, cr_encoded = diff_jpeg_coding(image_rgb=image_tensor, jpeg_quality=jpeg_quality, return_ycb=True)
    reward = -(torch.abs(image_tensor-image_coded)).mean(dim=[1, 2, 3])
    return reward


def light_reward():
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach()),{}
    return _fn


def jpeg_incompressibility():
    def _fn(images):
        if isinstance(images, torch.Tensor):
            if images.min() < 0:
                images = (images+1)/2
                images = images.clamp(0, 1) 
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        if not isinstance(images[0], Image.Image):    
            images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes)

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images):
        rew = jpeg_fn(images)
        return -rew

    return _fn


def clip_score(
    inference_dtype=torch.float32, 
    device=None, 
    return_loss=False, 
):
    from .clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    

def PickScore(
    inference_dtype=torch.float32, 
    device=None, 
    return_loss=False, 
):
    from .PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    

def aesthetic_score(
    inference_dtype=torch.float32,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    from .aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)

            if aesthetic_target is None: # default maximization
                loss = -1 * scores
            else:
                # using L1 to keep on same scale
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores

        return loss_fn


def hps_score(
    inference_dtype=torch.float32, 
    device=None, 
    return_loss=False, 
):
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn


def ImageReward(
    inference_dtype=torch.float32, 
    device=None, 
    return_loss=False, 
):
    from .ImageReward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=inference_dtype, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    
