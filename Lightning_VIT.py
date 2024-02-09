import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L


class EMGLightningNet(L.LightningModule):
# class EMGNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        # num of emg sensors
        self.num_channels = args.num_channels
        # num of fingers
        self.num_forces = args.num_forces
        # n=2 indicating the force is of or off
        self.num_force_levels = args.num_force_levels
        # number of time 
        self.num_frames = args.num_frames
        
        # STFT
        self.window_length = args.window_length
        self.hop_length = args.hop_length

        #### selected number of dimensions in frequency domain
        self.num_frequencies = args.num_frequencies
        # (batch, 8, 32, 64)
        
        # encoder
        self.encoder_channel = 32
        self.encoder1 = self._encoder_block(self.num_channels, self.encoder_channel, (1, 2))
        # (batch, 32, 32, 32)
        
        # transformer conv
        self.hidden_channel = 128
        self.patch_size = (4,4)
        #### convolution layer to segment image into patches
        self.patch_conv = nn.Conv2d(self.encoder_channel, self.hidden_channel, self.patch_size, stride=self.patch_size)
        # (batch, 128, 8,8)
        
        # vision transformer
        #### positional embedding
        self.positional_embeds = nn.Embedding(65, self.hidden_channel)
        #### positional embedding's id from 0 to a big number
        self.register_buffer("positional_ids", torch.arange(65).unsqueeze(0))
        #### number of transformer encoder layers
        self.num_layers = 6
        
        # (batch, 128, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.hidden_channel, # number of channels
                2, # number of heads in attention mechanism
                self.hidden_channel * 8 // 3, # the width of feedforward network
                activation=F.silu,
                batch_first=True,
                norm_first=True,
            ),
            self.num_layers,
        )
        # (batch, 128, 64)
        
        #### deconv back
        self.patch_deconv = nn.ConvTranspose2d(self.hidden_channel, self.hidden_channel//2, self.patch_size, stride=self.patch_size)
        # (batch, 64, 32, 32)
        
        # post encoder
        self.encoder2 = self._encoder_block(self.hidden_channel//2, self.hidden_channel//4, (1, 4))
        # (batch, 32, 32, 8)
        self.encoder3 = self._encoder_block(self.hidden_channel//4, self.hidden_channel//8, (1, 4))
        # (batch, 16, 32, 2)
        
        # (batch, 2, 32, 16)
        self.linear = nn.Linear(self.hidden_channel//8, self.num_forces)
        # (batch, 2, 32, 5)
        
        # optimizer
        self.lr = args.lr
        self.weight_decay = args.weight_decay
    
    # encoder block
    def _encoder_block(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, 
                       window=torch.hann_window(self.window_length, device=self.device),
                       center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True).abs()
        x = x.view(batch_size, self.num_channels, x.size(1), x.size(2))
        x = x.transpose(2, 3)[..., :self.num_frequencies]
        
        x = self.encoder1(x)
        hidden = self.patch_conv(x)
        n_seq = hidden.shape[2] * hidden.shape[3]
        pos_embeds = self.positional_embeds(self.positional_ids[:, :n_seq]).expand(hidden.shape[0], -1, -1)
        output = self.transformer(hidden.flatten(2).transpose(1, 2) + pos_embeds).transpose(-1, -2).view_as(hidden)
        x = self.patch_deconv(output)
        
        x = self.encoder2(x)
        x = self.encoder3(x)
        # (batch, 16, 32, 2)
        # x = x.view(x.size(0), x.size(1), x.size(2))

        x = x.transpose(1, 3)
        # (batch, 2, 32, 16)
        x = self.linear(x)
        # (batch, 2, 32, 5)
        x = x.transpose(2, 3)
        # x = x.view(x.size(0), self.num_force_levels, self.num_forces, x.size(2))
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        emg, force, force_class = batch
        logits = self.forward(emg)
        # Classification
        loss_classification = F.cross_entropy(logits, force_class)
        # Regression
        logits = logits.transpose(1, 3)
        probs = F.softmax(logits, 3)
        weights = torch.from_numpy(np.array([5], dtype=np.float32)).to(self.device)
        loss_regression = F.mse_loss((F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2), force)
        # Optimization
        loss = loss_classification + 4 * loss_regression
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        emg, force, force_class = batch
        logits = self.forward(emg)
        # Classification
        loss_classification = F.cross_entropy(logits, force_class)
        correct = logits.max(1)[1].eq(force_class).all(1).sum().item()
        all = force_class.shape[0] * force_class.shape[2]
        # Regression
        logits = logits.transpose(1, 3)
        probs = F.softmax(logits, 3)
        weights = torch.from_numpy(np.array([5], dtype=np.float32)).to(self.device)
        loss_regression = F.mse_loss((F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2), force)
        # Optimization
        loss = loss_classification + 4 * loss_regression
        accuracy = 100.0 * float(correct) / all
        self.log("valid_acc", accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    
class EMGLightningMLP(L.LightningModule):
# class EMGNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_channels = args.num_channels
        self.num_forces = args.num_forces
        self.num_force_levels = args.num_force_levels
        self.num_frames = args.num_frames
        self.num_frequencies = args.num_frequencies
        self.window_length = args.window_length
        self.hop_length = args.hop_length
        # STFT
        # hann_window = torch.hann_window(self.window_length, device=self.device)
        # # if args.cuda:
        # #     hann_window = hann_window.cuda()
        # self.hann_window = hann_window
        # Layers
        self.encoder = self._encoder()
        self.decoder = self._decoder()
        self.linear = nn.Linear(32, self.num_force_levels * self.num_forces)

        # optimizer
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        
    def _encoder(self):
        layers = []
        layers.append(self._encoder_block(self.num_channels, 32, (2, 4)))
        layers.append(self._encoder_block(32, 128, (2, 4)))
        layers.append(self._encoder_block(128, 256, (2, 4)))
        return nn.Sequential(*layers)

    def _encoder_block(self, in_channel, out_channel, downsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.MaxPool2d(kernel_size=downsample, stride=downsample))
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        layers.append(self._decoder_block(256, 256, (self.num_frames // 4, 1)))
        layers.append(self._decoder_block(256, 128, (self.num_frames // 2, 1)))
        layers.append(self._decoder_block(128, 32, (self.num_frames, 1)))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channel, out_channel, upsample):
        layers = []
        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding='same', bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Upsample(size=upsample, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_channels, x.size(2))
        x = torch.stft(x, n_fft=self.window_length, hop_length=self.hop_length, win_length=self.window_length, window=torch.hann_window(self.window_length, device=self.device),
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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        emg, force, force_class = batch
        logits = self.forward(emg)
        # Classification
        loss_classification = F.cross_entropy(logits, force_class)
        # Regression
        logits = logits.transpose(1, 3)
        probs = F.softmax(logits, 3)
        weights = torch.from_numpy(np.array([5], dtype=np.float32)).to(self.device)
        loss_regression = F.mse_loss((F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2), force)
        # Optimization
        loss = loss_classification + 4 * loss_regression
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        emg, force, force_class = batch
        logits = self.forward(emg)
        # Classification
        loss_classification = F.cross_entropy(logits, force_class)
        correct = logits.max(1)[1].eq(force_class).all(1).sum().item()
        all = force_class.shape[0] * force_class.shape[2]
        # Regression
        logits = logits.transpose(1, 3)
        probs = F.softmax(logits, 3)
        weights = torch.from_numpy(np.array([5], dtype=np.float32)).to(self.device)
        loss_regression = F.mse_loss((F.relu(probs[..., 1] - 0.5) * weights).transpose(1, 2), force)
        # Optimization
        loss = loss_classification + 4 * loss_regression
        accuracy = 100.0 * float(correct) / all
        self.log("valid_acc", accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

if __name__ == "__main__":
    pass
