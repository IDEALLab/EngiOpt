"""
File for implementing the Unet portion of the denoising diffusion model.
"""
from pdb import pm

from matplotlib.pylab import rint

from .cmpnts import time_embeddings, Up_Block, Middle_Block, Down_Block, AirfoilConditional, AirfoilConditionalPerf
from .utils import convert_str_to_activ

import torch
from torch import nn

class Unet_AoAInit(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self, 
        N_dim=30,
        latent_channels = 4, 
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        kernel_size = 3, 
        c_dim = 4, 
        c_dim_latent = 16,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.latent_channels = latent_channels 
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > latent_channels + c_dim_latent
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1

        self.alpha_dim = 1  # Alpha is a single channel

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x0_emb = nn.Parameter(torch.randn((latent_channels, N_dim)))  # Changed from latent_channels-1
        self.x_dim_add += latent_channels  # Changed from latent_channels-1 to latent_channels

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()

        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)
        if self.embed_c:
            time_emb_dim += c_dim_latent
        self.time_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        self.conv_f2 = nn.Conv1d(up_channels[-1], latent_channels + 1, kernel_size=1)  # +1 for alpha

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(latent_channels + c_dim_latent + self.x_dim_add + self.alpha_dim, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(1,1)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        x_channels = latent_channels 

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_channels, x_channels, kernel_size=1),
        )
        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
        print('latent_channels_full = ', 'latent_channels + ','c_dim_latent + ', 'x_dim_add + ', 'alpha_dim')
        print(str(self.latent_channels + self.c_dim_latent + self.x_dim_add + self.alpha_dim), ' = ', str(self.latent_channels), '+', str(self.c_dim_latent), '+', str(self.x_dim_add), '+', str(self.alpha_dim))
        print('down_channels[0] = ', down_channels[0])
        ### End of Parameters
        if down_channels[0] < self.latent_channels + self.c_dim_latent + self.x_dim_add:
            print('Warning: down_channels[0] < latent_channels + c_dim_latent + x_dim_add')
    
    def infer(self, x, alpha, c, x0, t):
        """
        Infer the output of the Unet
        """
        with torch.no_grad():
            return self.forward(x, alpha, c, x0, t)

    def forward(self, x, alpha, c, x0, t):
        """
        Forward pass of the Unet
        """
        # Pad the input:
        x = self.pad(x)
        
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        x[:, 2, -1] = x[:, 2, -1]*0
        x[:, 2, 0] = x[:, 2, 0]*0

        # Duplicate x0_multiple times:
        x0 = self.pad(x0)
        
        x0[:, 2, -1] = x0[:, 2, -1]*0
        x0[:, 2, 0] = x0[:, 2, 0]*0
        x0 = self.duplication_module(x0)
        
        # Concatenate x0 to x:
        x = torch.cat((x, x0), dim=1)

        if self.upsampling:
            x = self.upsampler(x)
        
        # Handle alpha - ensure it has the right shape
        if alpha.dim() == 2:  # [batch, 1]
            alpha = alpha.unsqueeze(1)  # [batch, 1, 1]
        elif alpha.dim() == 3 and alpha.shape[1] == 0:  # Handle empty case
            alpha = torch.zeros(alpha.shape[0], 1, x.shape[2]).to(x.device)
        
        # Repeat alpha to match spatial dimension
        if alpha.shape[-1] != x.shape[2]:
            alpha = alpha.repeat(1, 1, x.shape[2])
        
        x = torch.cat((x, alpha), dim=1)

        # Conditional Embedding:
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        # Time Embedding:
        t = self.time_embedder(t)
        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])
        
        x = torch.cat((x, c), dim=1)

        # First Convolution:
        x = self.conv0(x)
        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)  # Now outputs [batch, latent_channels+1, spatial_dim]

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Split into noise and alpha
        x_noise = x[:, :self.latent_channels, :]  # First latent_channels channels
        alpha = x[:, self.latent_channels:, :]    # Last channel

        # Take the mean of alpha across the spatial dimension
        alpha = torch.mean(alpha, dim=2)  # Shape: [batch, 1]

        # Remove padding from noise prediction
        x_noise = x_noise[:, :, 1:-1]

        return x_noise, alpha


class Unet_AoA(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self, 
        N_dim=30,
        latent_channels = 4, 
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        kernel_size = 3, 
        c_dim = 4, 
        c_dim_latent = 16,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.latent_channels = latent_channels 
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > latent_channels + c_dim_latent
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x0_emb = nn.Parameter(torch.randn((latent_channels-1, N_dim)))
        self.x_dim_add += (latent_channels-1)

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()

        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)
        if self.embed_c:
            time_emb_dim += c_dim_latent
        self.time_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        self.conv_f2 = nn.Conv1d(up_channels[-1], latent_channels, kernel_size=1)

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(latent_channels+c_dim_latent+self.x_dim_add, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(1,1)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        x_channels = latent_channels - 1

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_channels, x_channels, kernel_size=1),
        )
        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
    
        ### End of Parameters
    
    def infer(self, x, alpha, c, t):
        """
        Infer the output of the Unet
        """
        with torch.no_grad():
            return self.forward(x, alpha, c, t)

    def forward(self, x, alpha, c, t):
        """
        Forward pass of the Unet
        """
        # cond_drop_probs is a list of conditional dropout probabilities for each conditioning variable:
        # Ex: if c and x0 are active, cond_drop_probs = [0.1, 0.1], only c is active, cond_drop_probs = [0.1], etc.

        # Pad the input:
        x = self.pad(x)
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        x[:, 2, -1] = x[:, 2, -1]*0
        x[:, 2, 0] = x[:, 2, 0]*0

        if self.upsampling:
            x = self.upsampler(x)
        
        # Repeat alpha 
        alpha = alpha.unsqueeze(1).repeat(1, 1, x.shape[2])
        x = torch.cat((x, alpha), dim=1)

        # Conditional Embedding:
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        # Time Embedding:
        t = self.time_embedder(t)
        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, c), dim=1)

        # First Convolution:
        x = self.conv0(x)
        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Get alpha as the last channel:
        alpha = x[:, -1, :]
        # Take the mean of the last channel:
        alpha = torch.mean(alpha, dim=1).unsqueeze(1)
        x = x[:, :-1, :]

        # Remove padding:
        x = x[:, :, 1:-1]

        return x, alpha

class Unet_AoAInit_Corr(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self, 
        N_dim=30,
        latent_channels = 4, 
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        kernel_size = 3, 
        c_dim = 4, 
        c_dim_latent = 16,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        airfoil_cond_layers = [96, 64, 32],
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.latent_channels = latent_channels 
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > latent_channels + c_dim_latent
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x0_emb = nn.Parameter(torch.randn((latent_channels-1, N_dim)))
        self.x_dim_add += (latent_channels-1)

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()

        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)
        self.airfoil_cond_net = AirfoilConditional(3, N_dim, time_emb_dim, airfoil_cond_layers)

        if self.embed_c:
            time_emb_dim += c_dim_latent
        
        self.time_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        self.conv_f2 = nn.Conv1d(up_channels[-1], latent_channels, kernel_size=1)

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(latent_channels+c_dim_latent+self.x_dim_add, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(1,1)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        x_channels = latent_channels - 1

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_channels, x_channels, kernel_size=1),
        )
        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
    
        ### End of Parameters
    
    def infer(self, x, alpha, c, x0):
        """
        Infer the output of the Unet
        """
        with torch.no_grad():
            return self.forward(x, alpha, c, x0)

    def forward(self, x, alpha, c, x0):
        """
        Forward pass of the Unet
        """
        # cond_drop_probs is a list of conditional dropout probabilities for each conditioning variable:
        # Ex: if c and x0 are active, cond_drop_probs = [0.1, 0.1], only c is active, cond_drop_probs = [0.1], etc.
        t = self.airfoil_cond_net(x)
        
        # Pad the input:
        x = self.pad(x)
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        x[:, 2, -1] = x[:, 2, -1]*0
        x[:, 2, 0] = x[:, 2, 0]*0

        # Duplicate x0_multiple times:
        x0 = self.pad(x0)
        x0[:, 2, -1] = x0[:, 2, -1]*0
        x0[:, 2, 0] = x0[:, 2, 0]*0
        x0 = self.duplication_module(x0)
        # Concatenate x0 to x:
        x = torch.cat((x, x0), dim=1)

        if self.upsampling:
            x = self.upsampler(x)
        
        # Repeat alpha 
        alpha = alpha.unsqueeze(1).repeat(1, 1, x.shape[2])
        x = torch.cat((x, alpha), dim=1)

        # Conditional Embedding:
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, c), dim=1)

        # First Convolution:
        x = self.conv0(x)
        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Get alpha as the last channel:
        alpha = x[:, -1, :]
        # Take the mean of the last channel:
        alpha = torch.mean(alpha, dim=1).unsqueeze(1)
        x = x[:, :-1, :]

        # Remove padding:
        x = x[:, :, 1:-1]

        return x, alpha
    
class Unet_AoAInit_CorrPerf(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self, 
        N_dim=30,
        latent_channels = 4, 
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        kernel_size = 3, 
        c_dim = 4, 
        c_dim_latent = 16,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        perf_dim = 2,
        airfoil_perf_net = [96, 64, 32],
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.latent_channels = latent_channels 
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > latent_channels + c_dim_latent
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x0_emb = nn.Parameter(torch.randn((latent_channels-1, N_dim)))
        self.x_dim_add += (latent_channels-1)

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()

        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)

        self.airfoil_perf_net = AirfoilConditionalPerf(3, N_dim, 2, perf_dim, airfoil_perf_net)


        if self.embed_c:
            perf_dim += c_dim_latent
        
        self.time_net = nn.Sequential(
            nn.Linear(perf_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        self.conv_f2 = nn.Conv1d(up_channels[-1], latent_channels, kernel_size=1)

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(latent_channels+c_dim_latent+self.x_dim_add, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(1,1)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        x_channels = latent_channels - 1

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_channels, x_channels, kernel_size=1),
        )
        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
    
        ### End of Parameters
    
    def infer(self, x, alpha, c, x0):
        """
        Infer the output of the Unet
        """
        with torch.no_grad():
            x, alpha, _ = self.forward(x, alpha, c, x0)
            return x, alpha

    def forward(self, x, alpha, c, x0):
        """
        Forward pass of the Unet
        """
        # cond_drop_probs is a list of conditional dropout probabilities for each conditioning variable:
        # Ex: if c and x0 are active, cond_drop_probs = [0.1, 0.1], only c is active, cond_drop_probs = [0.1], etc.
        # MRe is the first two dimensions of c
        MRe = c[:, :2]
        performance = self.airfoil_perf_net(x, MRe, alpha)
        t = performance.clone()

        # Pad the input:
        x = self.pad(x)
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        x[:, 2, -1] = x[:, 2, -1]*0
        x[:, 2, 0] = x[:, 2, 0]*0

        # Duplicate x0_multiple times:
        x0 = self.pad(x0)
        x0[:, 2, -1] = x0[:, 2, -1]*0
        x0[:, 2, 0] = x0[:, 2, 0]*0
        x0 = self.duplication_module(x0)
        # Concatenate x0 to x:
        x = torch.cat((x, x0), dim=1)

        if self.upsampling:
            x = self.upsampler(x)
        
        # Repeat alpha 
        alpha = alpha.unsqueeze(1).repeat(1, 1, x.shape[2])
        x = torch.cat((x, alpha), dim=1)

        # Conditional Embedding:
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, c), dim=1)

        # First Convolution:
        x = self.conv0(x)
        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Get alpha as the last channel:
        alpha = x[:, -1, :]
        # Take the mean of the last channel:
        alpha = torch.mean(alpha, dim=1).unsqueeze(1)
        x = x[:, :-1, :]

        # Remove padding:
        x = x[:, :, 1:-1]

        return x, alpha, performance

class Unet_AoAInitTraj(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self, 
        N_dim=30,
        latent_channels = 4, 
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        kernel_size = 3, 
        c_dim = 4, 
        c_dim_latent = 16,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.latent_channels = latent_channels 
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > latent_channels + c_dim_latent
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x0_emb = nn.Parameter(torch.randn((latent_channels-1, N_dim)))
        self.x_dim_add += (latent_channels-1)

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()
        c_dim += 1
        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)
        if self.embed_c:
            time_emb_dim += c_dim_latent
        self.time_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        self.conv_f2 = nn.Conv1d(up_channels[-1], latent_channels, kernel_size=1)

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(latent_channels+c_dim_latent+self.x_dim_add, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(1,1)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        x_channels = latent_channels - 1

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_channels, x_channels, kernel_size=1),
        )
        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
    
        ### End of Parameters
    
    def infer(self, x, alpha, c, x0, t, pct=None):
        """
        Infer the output of the Unet
        """
        with torch.no_grad():
            if pct is None:
                pct = torch.zeros(x.shape[0], 1)
            return self.forward(x, alpha, c, x0, t, pct)

    def forward(self, x, alpha, c, x0, t, pct):
        """
        Forward pass of the Unet
        """
        # cond_drop_probs is a list of conditional dropout probabilities for each conditioning variable:
        # Ex: if c and x0 are active, cond_drop_probs = [0.1, 0.1], only c is active, cond_drop_probs = [0.1], etc.

        # Pad the input:
        x = self.pad(x)
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        x[:, 2, -1] = x[:, 2, -1]*0
        x[:, 2, 0] = x[:, 2, 0]*0

        # Duplicate x0_multiple times:
        x0 = self.pad(x0)
        x0[:, 2, -1] = x0[:, 2, -1]*0
        x0[:, 2, 0] = x0[:, 2, 0]*0
        x0 = self.duplication_module(x0)
        # Concatenate x0 to x:
        x = torch.cat((x, x0), dim=1)

        if self.upsampling:
            x = self.upsampler(x)
        
        # Repeat alpha 
        alpha = alpha.unsqueeze(1).repeat(1, 1, x.shape[2])
        x = torch.cat((x, alpha), dim=1)

        # Conditional Embedding:
        c = torch.cat((c, pct), dim=1)
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        # Time Embedding:
        t = self.time_embedder(t)
        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, c), dim=1)

        # First Convolution:
        x = self.conv0(x)
        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Get alpha as the last channel:
        alpha = x[:, -1, :]
        # Take the mean of the last channel:
        alpha = torch.mean(alpha, dim=1).unsqueeze(1)
        x = x[:, :-1, :]

        # Remove padding:
        x = x[:, :, 1:-1]

        return x, alpha

class Unet_AoAInit3D(nn.Module):
    """
    Unet archetecture using 1D convolutions
    """
    def __init__(
        self,                 
        w_dim = 9,
        x_latent_channels_2D = 3, # x latent + c latent of size x latent concatenated with c latent
        N_dim = 30,
        tform_dim = 1,
        down_channels = [16, 32, 64, 128], 
        middle_channel = 64, 
        up_channels = [128, 64, 32, 16],
        time_emb_dim = 64, 
        upsampling_factor = 1,
        pad_size = 1,
        kernel_size = 3, 
        alpha_dim_latent = 3,
        x02D_dim_latent_multiply_factor = 1, 
        c_dim = 4, 
        c_dim_latent = 10,
        c_net_hidden_layers = [16, 16],
        c_net_activation = 'GELU', 
        c_net_last_activation = 'GELU', 
        time_net_activation = 'GELU', 
        time_net_last_activation = 'GELU',
        block_activation = 'GELU', 
        padding_mode = 'circular', 
        block_norms = [False, False, False],
        embed_c = True,
        droput = True,
        dropout_prob = 0.3,
        ):
        super().__init__()
        ### Parameters:

        # Activations:
        c_net_activation = convert_str_to_activ(c_net_activation)
        c_net_last_activation = convert_str_to_activ(c_net_last_activation)
        time_net_activation = convert_str_to_activ(time_net_activation)
        time_net_last_activation = convert_str_to_activ(time_net_last_activation)
        block_activation = convert_str_to_activ(block_activation)

        # Size Parameters:
        self.w_dim = w_dim
        self.x_latent_channels = x_latent_channels_2D 
        self.N_dim = N_dim
        self.tform_dim = tform_dim
        self.down_channels = down_channels
        # Note: probably a good idea to keep down_channels[0] > self.latent_channels_init
        self.middle_channel = middle_channel
        # Note: the last up channel does not necessarily have to be the same as the first down channel
        self.up_channels = up_channels # must have minimum length of 2
        self.time_emb_dim = time_emb_dim
        self.upsampling_factor = upsampling_factor
        self.kernel_size = kernel_size
        self.upsampling = upsampling_factor > 1
        self.alpha_dim_latent = alpha_dim_latent
        self.x02D_dim_latent_multiply_factor = x02D_dim_latent_multiply_factor
        
        #Embedding:
        self.span_pos_emb = nn.Parameter(torch.randn(w_dim, x_latent_channels_2D))

        # Conditional Parameters:
        self.c_dim = c_dim # Ex: Mach, Re, Cl
        self.c_dim_latent = c_dim_latent # Number of dimensions to scale the conditional embedding to

        # x0 condition (condition on an additional/initial airfoil shape)
        self.x_latent_channels_3D = w_dim * x_latent_channels_2D
        self.x_dim_add = 0
        self.null_c_emb = nn.Parameter(torch.randn(c_dim))
        self.null_x02D_emb = nn.Parameter(torch.randn((x_latent_channels_2D, N_dim)))
        self.x02d_dim_latent = x_latent_channels_2D * x02D_dim_latent_multiply_factor
        self.latent_channels_init = self.x_latent_channels_3D + self.c_dim_latent + self.alpha_dim_latent + self.x02d_dim_latent

        ## Activation and Network Components:

        # Conditional Latent:
        self.c_net_activation = c_net_activation
        self.c_net__last_activation = c_net_last_activation 
        self.c_net_hidden_layers = c_net_hidden_layers
        self.c_net = nn.ModuleList()
        self.embed_c = embed_c

        self.droput = droput
        self.dropout_prob = dropout_prob
        # outputs size c_dim_latent
        if c_net_last_activation is None:
            self.c_net__last_activation = nn.Identity()

        self.c_net.append(nn.Sequential(nn.Linear(c_dim, c_net_hidden_layers[0]), c_net_activation))
        for i in range(len(c_net_hidden_layers)-1):
            self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[i], c_net_hidden_layers[i+1]), c_net_activation))
        self.c_net.append(nn.Sequential(nn.Linear(c_net_hidden_layers[-1], c_dim_latent), c_net_last_activation))  

        # Time Embedding:
        self.time_net_activation = time_net_activation
        self.time_net_last_activation = time_net_last_activation
        self.time_embedder = time_embeddings(L=time_emb_dim)
        if self.embed_c:
            time_emb_dim += c_dim_latent
        self.time_net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_activation,
            nn.Linear(time_emb_dim, time_emb_dim),
            time_net_last_activation
        )

        # Down/Middle/Up Blocks:
        self.block_activation = block_activation # Activation function for the down/middle/up blocks
        pm = padding_mode # Padding mode for the down/middle/up blocks
        self.block_norms = block_norms # Whether or not to use batch norm for the down/middle/up blocks
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.down_blocks.append(Down_Block(down_channels[i], down_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                                padding_mode=pm, activation=block_activation, norm=block_norms[0]))
        
        self.middle_block = Middle_Block(down_channels[-1], middle_channel, up_channels[0], time_emb_dim, kernel_size=kernel_size,\
                                         padding_mode=pm, activation=block_activation, norm=block_norms[1])

        self.up_blocks = nn.ModuleList()

        for i in range(len(up_channels)-1):
            if i == len(up_channels)-2:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i], time_emb_dim, kernel_size=kernel_size,\
                                               padding_mode=pm, activation=block_activation, norm=block_norms[2], final=True))
            else:
                self.up_blocks.append(Up_Block(up_channels[i], up_channels[i+1], time_emb_dim, kernel_size=kernel_size,\
                                           padding_mode=pm, activation=block_activation, norm=block_norms[2], final=False))

        # Final Block:
        self.conv_f1_combo = nn.Sequential(
             nn.Conv1d(up_channels[-2], up_channels[-1], kernel_size=1),
             block_activation
        )
        
        # Output channels: geometry (x_latent_channels_3D) + AoA (1)
        self.conv_f2 = nn.Conv1d(up_channels[-1], self.x_latent_channels_3D + 1, kernel_size=1)

        self.avg_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Misc:
        self.conv0 = nn.Conv1d(self.latent_channels_init, down_channels[0], kernel_size=3, padding=1, padding_mode=pm)
        self.pad = nn.ConstantPad1d(pad_size,pad_size)
        self.upsampler = nn.Upsample(scale_factor=upsampling_factor, mode='nearest')

        # Create a duplication module
        # Creates x0_multiple copies of the input channel

        self.duplication_module = nn.Sequential(
            nn.Conv1d(x_latent_channels_2D, x_latent_channels_2D*self.x02D_dim_latent_multiply_factor, kernel_size=1),
        )

        self.alpha_duplication_module = nn.Sequential(
            nn.Conv1d(1, self.alpha_dim_latent, kernel_size=1),
        )

        ### End of Parameters
        # Warn if down_channels[0] < self.latent_channels_init
        
        print("Number of latent channels: ", self.latent_channels_init)
        if down_channels[0] < self.latent_channels_init:
            print('Warning: down_channels[0] < self.latent_channels_init.')
            print('latent_channels = x_latent_channels_3D + ' + 'c_dim_latent + ' + 'alpha_dim_latent + ' + 'x02d_dim_latent')
            print(str(self.latent_channels_init) + ' = ' + str(self.x_latent_channels_3D) + ' + ' + str(self.c_dim_latent) + ' + ' + str(self.alpha_dim_latent) + ' + ' + str(self.x02d_dim_latent))
    
    def infer(self, x, alpha, c, x0, t, pct=None):
        """
        Infer the output of the Unet.
        Returns (x_noise_pred, alpha_noise_pred).
        """
        with torch.no_grad():
            if pct is None:
                pct = torch.zeros(x.shape[0], 1)
            return self.forward(x, alpha, c, x0, t)

    def forward(self, x:torch.Tensor, alpha:torch.Tensor, c:torch.Tensor, x0:torch.Tensor, t:torch.Tensor, debug=False):
        """
        Forward pass of the Unet
        """
        # cond_drop_probs is a list of conditional dropout probabilities for each conditioning variable:
        # Ex: if c and x0 are active, cond_drop_probs = [0.1, 0.1], only c is active, cond_drop_probs = [0.1], etc.
        if debug:
            print('(1)>>> x starting shape: ', x.shape)

        if self.droput:
            # Apply dropout to x
            # x = torch.nn.functional.dropout(x, p=self.dropout_prob, training=self.training)
            c = torch.nn.functional.dropout(c, p=self.dropout_prob, training=self.training)
            # x0 = torch.nn.functional.dropout(x0, p=self.dropout_prob, training=self.training)
            # alpha = torch.nn.functional.dropout(alpha, p=self.dropout_prob, training=self.training)
        
        # Reshape x from [B, w_dim, 3, N] to [B, w_dim*3, N]
        # pad 0, 0 for y, 1,1 for w, and x; this way we mimic exactly the trailing edge condition
        # Pad the input:
        x = self.pad(x)
        if debug:
            print('(2)>>> x padded shape: ', x.shape)
        x[:, :, 2, -1] = x[:, :, 2, -1]*0
        x[:, :, 2, 0] = x[:, :, 2, 0]*0
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if debug:
            print('(3)>>> x reshaped shape: ', x.shape)

        # Duplicate x0_multiple times:
        x0 = self.pad(x0)
        x0[:, 2, -1] = x0[:, 2, -1]*0
        x0[:, 2, 0] = x0[:, 2, 0]*0
        x0 = self.duplication_module(x0)
        if debug:
            print('(4)>>> x0 duplicated shape: ', x0.shape)
        # Concatenate x0 to x:
        x = torch.cat((x, x0), dim=1)
        if debug:
            print('(5)>>> x concatenated with x0 shape: ', x.shape)

        if self.upsampling:
            x = self.upsampler(x)
    
        if debug:
            print('(6)>>> x upsampled shape: ', x.shape)

        # Repeat alpha 
        alpha = alpha.unsqueeze(1).repeat(1, 1, x.shape[2])

        if debug:
            print('(7)>>> alpha repeated shape: ', alpha.shape)

        alpha = self.alpha_duplication_module(alpha)

        if debug:
            print('(8)>>> alpha duplicated shape: ', alpha.shape)

        x = torch.cat((x, alpha), dim=1)

        if debug:
            print('(9)>>> x concatenated with alpha shape: ', x.shape)
            print('(10)>>> c shape: ', c.shape)

        # Conditional Embedding:
        for i in range(len(self.c_net)):
            c = self.c_net[i](c)

        if debug:
            print('(11)>>> c after c_net shape: ', c.shape)

        # Time Embedding:
        t = self.time_embedder(t)
        if self.embed_c:
            t = torch.cat((t, c), dim=1)
        t = self.time_net(t)
        
        c = c.unsqueeze(2).repeat(1, 1, x.shape[2])

        if debug:
            print('(12)>>> c repeated shape: ', c.shape)

        x = torch.cat((x, c), dim=1)

        if debug:
            print('(13)>>> x concatenated with c shape: ', x.shape)

        # First Convolution:
        x = self.conv0(x)

        if debug:
            print('(14)>>> x after conv0 shape: ', x.shape)

        res = []

        # Down Blocks:
        for i in range(len(self.down_blocks)):
            x, r = self.down_blocks[i](x, t)
            res.append(r)
        
        # Middle Block:
        x = self.middle_block(x, t)

        # Up Blocks:
        for i in range(len(self.up_blocks)):
            res_cat = res.pop()
            x = torch.cat((x, res_cat), dim=1)
            x = self.up_blocks[i](x, t)
        
        # Final Block:
        x = self.conv_f1_combo(x)
        x = self.conv_f2(x)

        # Pooling to undo upsampling:
        if self.upsampling:
            x = self.avg_pool(x)

        # Split output channels: [...geometry... | alpha (1)]
        # alpha: last channel, averaged over spatial dim  -> [B, 1]
        alpha = torch.mean(x[:, -1, :], dim=1).unsqueeze(1)
        # x: first x_latent_channels_3D channels
        x = x[:, :-1, :]

        # Remove padding:
        x = x[:, :, 1:-1]

        # Reshape x from [B, w_dim*3, N] to [B, w_dim, 3, N]
        x = x.reshape(x.shape[0], self.w_dim, 3, x.shape[-1])

        span_emb = self.span_pos_emb.unsqueeze(0).unsqueeze(-1)  # [1, w_dim, 3, 1]
        span_emb = span_emb.expand(x.shape[0], -1, -1, x.shape[-1])  # [B, w_dim, 3, N]
        x = x + span_emb 

        return x, alpha