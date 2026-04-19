import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelCountColorizer(nn.Module):
    # Recolor each channel's active pixels to the color == pixel count (clamped to 9)
    def forward(self, x):
        # x is (1, 10, H, W) one-hot
        counts = x.sum(dim=(2, 3), keepdim=True) # (1, 10, 1, 1)
        target_colors = torch.clamp(counts, 0, 9).long() # (1, 10, 1, 1)
        
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        
        # We want to place 1.0 in `out` at channel c' for every pixel that is 1.0 in x's channel c, where c' = target_colors[0, c, 0, 0]
        # This is equivalent to multiplying x by one-hot(target_colors) and summing.
        target_one_hot = F.one_hot(target_colors.squeeze(0).squeeze(1).squeeze(1), num_classes=10).transpose(0, 1)
        # target_one_hot is (10, 10) where target_one_hot[c', c] = 1 if channel c maps to color c'
        
        # x is (B, C, H*W). We want output B, C, H*W
        x_flat = x.view(B, C, -1)
        # target_one_hot: (10, 10) wait, F.one_hot output is float? No, int.
        out_flat = torch.matmul(target_one_hot.float().unsqueeze(0), x_flat)
        return out_flat.view(B, C, H, W)

m = ChannelCountColorizer()
dummy = torch.zeros(1, 10, 4, 4)
dummy[0, 1, 0:2, 0:2] = 1 # count = 4
dummy[0, 2, 2:4, 2:4] = 1 # count = 4

res = m(dummy)
print(res[0, 4, :, :]) # should be 1 where dummy had 1s!
