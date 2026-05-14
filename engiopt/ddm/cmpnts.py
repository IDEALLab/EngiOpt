"""
File for implementing components of the denoising diffusion model and related classes.
    Contains:
        - time_embeddings class
        - Up_Block class
        - Middle_Block class
        - Down_Block class
"""

import math

import torch
from torch import nn

class time_embeddings(nn.Module):
    def __init__(self, L, n=10000):
        super().__init__()
        self.L = L
        self.d = self.L // 2
        self.n = n

    def forward(self, t: torch.Tensor):
        # Compute the positional encodings once in log space.
        # PE(k, 2i) = sin(k / n^(2i/d))
        # PE(k, 2i+1) = cos(k / n^(2i/d))
        # L: length of the sequence
        # k: position of object in sequence from 0 to L/2
        # d: output dimension of the positional encoding
        # i: dimension of the positional encoding
        # n: scaling factor

        # Compute the positional encodings once in log space.
        # column position
        col_id = torch.arange(self.d, dtype=torch.float, device=t.device)
        # e.g: cos( k/n^(i/d) ) = cos(k / exp( -ln(n) i / d) )
        pe = t.unsqueeze(-1) * torch.exp(-math.log(self.n) * col_id / self.d).unsqueeze(0)
        # concatenate the sin and cos functions
        pe = torch.cat((torch.sin(pe), torch.cos(pe)), dim=-1)
        
        return pe

class Up_Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, kernel_size=3, 
                 padding=1, padding_mode='circular', activation= nn.GELU(), norm=True, final=False):
        super().__init__()


        self.conv_u1_combo = nn.Sequential(
            nn.Conv1d(2*in_channel, in_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation,
        )
        self.conv_u2_combo = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation,
        )
        self.conv_u3 = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2)

        self.time_resizer = nn.Sequential(nn.Linear(time_dim, in_channel), activation)
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(in_channel)
        self.norm2 = nn.BatchNorm1d(in_channel)
        self.final = final

    def forward(self, x, t):
        x = self.conv_u1_combo(x)
        if self.norm:
            x = self.norm1(x)
        
        t = self.time_resizer(t)
        t = t.unsqueeze(-1)
        x = x + t

        x = self.conv_u2_combo(x)
        if self.norm:
            x = self.norm2(x)
        
        if not self.final:
            x = self.conv_u3(x)

        return x

class Middle_Block(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel, time_dim, kernel_size=3,
                 padding=1, padding_mode='circular', activation=nn.GELU(), norm=True):
        super().__init__()
        self.conv_m1_combo = nn.Sequential(
            nn.Conv1d(in_channel, middle_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation
        )
        self.conv_m2_combo = nn.Sequential(
            nn.Conv1d(middle_channel, middle_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation
        )

        self.conv_m3 = nn.ConvTranspose1d(middle_channel, out_channel, kernel_size=2, stride=2)

        self.time_resizer = nn.Sequential(nn.Linear(time_dim, middle_channel), activation)
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(middle_channel)
        self.norm2 = nn.BatchNorm1d(middle_channel)
        
    def forward(self, x, t):
        x = self.conv_m1_combo(x)
        if self.norm:
            x = self.norm1(x)
        x = self.conv_m2_combo(x)
        if self.norm:
            x = self.norm2(x)

        t = self.time_resizer(t)
        t = t.unsqueeze(-1)
        x = x + t
        x = self.conv_m3(x)

        return x    

class Down_Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, kernel_size=3,
                 padding=1, padding_mode='circular', activation=nn.GELU(), norm=True):
        super().__init__()
        self.conv_d1_combo = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation
        )
        self.conv_d2_combo = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
            activation
        ) 
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.time_resizer = nn.Sequential(nn.Linear(time_dim, out_channel), activation)
        self.norm = norm
        self.norm1 = nn.BatchNorm1d(out_channel)
        self.norm2 = nn.BatchNorm1d(out_channel)

    def forward(self, x, t):

        x = self.conv_d1_combo(x)
        if self.norm:
            x = self.norm1(x)

        t = self.time_resizer(t)
        t = t.unsqueeze(-1)
        x = x + t

        x = self.conv_d2_combo(x)
        if self.norm:
            x = self.norm2(x)

        # returns x, res
        return self.pool(x), x
    
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, in_channel, out_channel, n_coefficients, activation=nn.GELU(), norm=True):
        """
        :param in_channel: number of feature maps (channels) in previous layer
        :param out_channel: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()
        self.activation = activation
        self.norm = norm

        self.W_gate = nn.Sequential(
            nn.Conv1d(in_channel, n_coefficients, padding=0, kernel_size=1, stride=1, bias=True),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(out_channel, n_coefficients, padding=0, kernel_size=1, stride=1, bias=True),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.norm:
            self.W_gate.add_module('norm', nn.BatchNorm1d(out_channel))
            self.W_x.add_module('norm', nn.BatchNorm1d(out_channel))
            self.psi.add_module('norm', nn.BatchNorm1d(1))
        self.psi.add_module('sigmoid', nn.Sigmoid())

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, input):
        return self.model(input)

class LinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha)
        )
class LinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha)
        )

class MLP(nn.Module):
    """Regular fully connected network generating features.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        layer_width: The widths of the hidden layers.
        combo: The layer combination to be stacked up.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, 
        combo = LinearCombo
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, combo)
    
    def forward(self, input):
        return self.model(input)

    def _build_model(self, layer_width, combo):
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip(
            [self.in_features] + layer_width, 
            layer_width + [self.out_features]
            )):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        return model
class AirfoilConditional(nn.Module):
    """Takes in Bezier airfoil control points and weights, 
    and outputs a 1D tensor of conditional parameters with a chosen size.
    """
    def __init__(self, in_channels: int, in_dim: int, out_features: int = 10, mlp_layers: list = [96, 64, 32]):
        super().__init__()
        self.mlp = MLP(in_dim*in_channels, out_features, layer_width=mlp_layers)
        self.flatten = nn.Flatten(start_dim=1)
    def forward(self, wcp: torch.Tensor):
        # Flatten the control points and weights
        x = self.flatten(wcp)
        return self.mlp(x)

class AirfoilConditionalPerf(nn.Module):
    """Takes in Bezier airfoil control points and weights, 
    and outputs a 1D tensor of conditional parameters with a chosen size.
    """
    def __init__(self, in_channels: int, in_dim: int, inp_paras_dim: int, out_features: int = 10, mlp_layers: list = [96, 64, 32]):
        super().__init__()
        self.mlp = MLP(in_dim*in_channels + inp_paras_dim + 1, out_features, layer_width=mlp_layers)
        self.flatten = nn.Flatten(start_dim=1)
    def forward(self, wcp: torch.Tensor, inp_paras: torch.Tensor, alpha: torch.Tensor):
        # Flatten the control points and weights
        x = self.flatten(wcp)
        x = torch.cat((x, inp_paras, alpha), dim=1)
        return self.mlp(x)