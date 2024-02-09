import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


class EMGLightningNet(L.LightningModule):

    def __init__(self, args):
        super(EMGNet, self).__init__()
        # num of emg sensors
        self.num_channels = args.num_channels
        # num of fingers
        self.num_forces = args.num_forces
        # n=2 indicating the force is of or off
        self.num_force_levels = args.num_force_levels
        # number of time 
        self.num_frames = args.num_frames
        
        # transformer
        
        # STFT
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.hann_window = torch.hann_window(self.window_length)
        #### selected number of dimensions in frequency domain
        self.num_frequencies = args.num_frequencies
        
        # vision transformer
        #### positional embedding
        self.positional_embeds = nn.Embedding(100000, hidden_size)
        #### positional embedding's id from 0 to a big number
        self.register_buffer("positional_ids", torch.arange(100000).unsqueeze(0))
        ####
        self.patch_conv = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        #### 
        self.patch_deconv = nn.ConvTranspose2d(hidden_size, out_channels, patch_size, stride=patch_size)

        # transfer the shape of output into self.num_force_levels * self.num_forces
        self.linear = nn.Linear(32, self.num_force_levels * self.num_forces)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=self.hann_window,
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), self.num_force_levels, self.num_forces, x.size(2))
        return x
        

from torch import nn
import torch
import torch.nn.functional as F


class VisionTransformer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, patch_size: int, num_layers: int):
        super().__init__()
        self.positional_embeds = nn.Embedding(100000, hidden_size)
        self.register_buffer("positional_ids", torch.arange(100000).unsqueeze(0))
        self.patch_conv = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.patch_deconv = nn.ConvTranspose2d(hidden_size, out_channels, patch_size, stride=patch_size)
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size,
                hidden_size // 64,
                hidden_size * 8 // 3,
                activation=F.silu,
                batch_first=True,
                norm_first=True,
            ),
            num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_conv(x)
        n_seq = hidden.shape[2] * hidden.shape[3]
        pos_embeds = self.positional_embeds(self.positional_ids[:, :n_seq]).expand(hidden.shape[0], -1, -1)
        output = self.model(hidden.flatten(2).transpose(1, 2) + pos_embeds).transpose(-1, -2).view_as(hidden)
        output = self.patch_deconv(output)
        return output


if __name__ == "__main__":
    pass
