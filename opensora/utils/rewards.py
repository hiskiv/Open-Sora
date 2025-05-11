import torch

def video_normalize(video):
    """ video shape: [B, C, T, H, W] """
    low, high = (-1, 1)
    x.clamp_(min=low, max=high)
    x.sub_(low).div_(max(high - low, 1e-5))

    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1) # [B, T, H, W, C]

    return x

def circle_reward(video_gt, video_gen):
    x = video_normalize(video_gen) # [B, T, H, W, C]

    