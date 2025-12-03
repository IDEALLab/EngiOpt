"""PixelCNN++ 2D model implementation.

Provides the model classes, shifted convolutional blocks, and the
discretized mixture of logistics loss used for training and sampling.
"""
from dataclasses import dataclass
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import tqdm
import tyro
import wandb


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 4
    """number of epochs of training"""
    sample_interval: int = 500
    """interval between image samples"""
    model_storage_interval: int = 2
    """interval between model storage"""
    batch_size: int = 8
    """size of the batches"""
    lr: float = 0.001
    """learning rate"""
    b1: float = 0.95
    """decay of first order momentum of gradient"""
    b2: float = 0.9995
    """decay of first order momentum of gradient"""
    nr_resnet: int = 2
    """Number of residual blocks per stage of the model."""
    nr_filters: int = 40
    """Number of filters to use across the model. Higher = larger model."""
    nr_logistic_mix: int = 5
    """Number of logistic components in the mixture. Higher = more flexible model."""
    resnet_nonlinearity: str = "concat_elu"
    """Nonlinearity to use in the ResNet blocks. One of 'concat_elu', 'elu', 'relu'."""
    dropout_p: float = 0.5
    """Dropout probability."""


def concat_elu(x):
    """Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU."""
    # Pytorch ordering
    axis = len(x.size()) - 3
    return F.elu(th.cat([x, -x], dim=axis))

class nin(nn.Module):
    def __init__(self, nr_filters_in, nr_filters_out):
        super().__init__()
        self.lin_a = weight_norm(nn.Linear(nr_filters_in, nr_filters_out))
        self.nr_filters_out = nr_filters_out

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        xs = list(x.shape)
        x = x.reshape(-1, xs[3])  # -> [B*H*W, C]
        out = self.lin_a(x)
        out = out.view(xs[0], xs[1], xs[2], self.nr_filters_out)
        return out.permute(0, 3, 1, 2)  # BHWC -> BCHW

class GatedResnet(nn.Module):
    def __init__(self, nr_filters, conv_op, resnet_nonlinearity=concat_elu, skip_connection=0, dropout_p=0.5):
        super().__init__()
        self.skip_connection = skip_connection
        self.resnet_nonlinearity = resnet_nonlinearity

        if resnet_nonlinearity is concat_elu:
            self.filter_doubling = 2
        else:
            self.filter_doubling = 1

        self.conv_input = conv_op(self.filter_doubling * nr_filters, nr_filters)

        if skip_connection != 0:
            self.nin_skip = nin(self.filter_doubling * skip_connection * nr_filters, nr_filters)

        self.dropout = nn.Dropout2d(dropout_p)
        self.conv_out = conv_op(self.filter_doubling * nr_filters, 2 * nr_filters) # output has to be doubled for gating

        self.h_lin = nn.Linear(nr_conditions, 2 * nr_filters)


    def forward(self, x, a=None, h=None):
        c1 = self.conv_input(self.resnet_nonlinearity(x))
        # print(f"c1 shape: {c1.shape}")
        if a is not None :
            # print(f"a shape: {a.shape}")
            c1 += self.nin_skip(self.resnet_nonlinearity(a))
        c1 = self.resnet_nonlinearity(c1)
        c1 = self.dropout(c1)
        c2 = self.conv_out(c1)
        if h is not None:
            # in forward, when `h` is [B, nr_conditions, 1, 1]
            h_flat = h.view(h.size(0), -1)                        # [B, nr_conditions]
            h_proj = self.h_lin(h_flat).unsqueeze(-1).unsqueeze(-1)  # [B, 2*nr_filters, 1, 1]
            c2 += h_proj
        a, b = th.chunk(c2, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return x + c3


def downShift(x, pad):
    xs = list(x.shape)
    x = x[:, :, :xs[2] - 1, :]
    x = pad(x)
    return x

def downRightShift(x, pad):
    xs = list(x.shape)
    x = x[:, :, :, :xs[3] - 1]
    x = pad(x)
    return x

class DownShiftedConv2d(nn.Module):
    def __init__(self,
                 nr_filters_in,
                 nr_filters_out,
                 filter_size=(2,3),
                 stride=(1,1),
                 shift_output_down=False):

        super().__init__()
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2), filter_size[0]-1, 0)) # padding left, right, top, bottom
        self.conv = weight_norm(nn.Conv2d(nr_filters_in, nr_filters_out, filter_size, stride))
        self.shift_output_down = shift_output_down
        self.down_shift = downShift
        self.down_shift_pad = nn.ZeroPad2d((0,0,1,0))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.shift_output_down:
            x = self.down_shift(x, pad=self.down_shift_pad)
        return x

class DownShiftedDeconv2d(nn.Module):
    def __init__(self,
                 nr_filters_in,
                 nr_filters_out,
                 filter_size=(2,3),
                 stride=(1,1),
                 output_padding=(0,1)):

        super().__init__()
        self.deconv = weight_norm(nn.ConvTranspose2d(nr_filters_in, nr_filters_out, filter_size, stride, output_padding=output_padding))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x, output_padding=None):
        # Allow callers to pass a dynamic `output_padding` (some callers compute
        # this to handle odd/even spatial sizes). If not provided, use the
        # configured ConvTranspose2d module directly.
        if output_padding is None:
            x = self.deconv(x)
        else:
            x = F.conv_transpose2d(
                x,
                self.deconv.weight,
                self.deconv.bias,
                stride=self.stride,
                padding=self.deconv.padding,
                output_padding=output_padding,
                dilation=self.deconv.dilation,
                groups=self.deconv.groups,
            )
        xs = list(x.shape)
        return x[:, :, :(xs[2] - self.filter_size[0] + 1), int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]

class DownRightShiftedConv2d(nn.Module):
    def __init__(self,
                 nr_filters_in,
                 nr_filters_out,
                 filter_size=(2,2),
                 stride=(1,1),
                 shift_output_right_down=False):

        super().__init__()
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0]-1, 0)) # padding left, right, top, bottom
        self.conv = weight_norm(nn.Conv2d(nr_filters_in, nr_filters_out, filter_size, stride))
        self.shift_output_right_down = shift_output_right_down
        self.down_right_shift = downRightShift
        self.down_right_shift_pad = nn.ZeroPad2d((1,0,0,0))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.shift_output_right_down:
            x = self.down_right_shift(x, pad=self.down_right_shift_pad)
        return x

class DownRightShiftedDeconv2d(nn.Module):
    def __init__(self,
                 nr_filters_in,
                 nr_filters_out,
                 filter_size=(2,2),
                 stride=(1,1),
                 output_padding=(1,0)):

        super().__init__()
        self.deconv = weight_norm(nn.ConvTranspose2d(nr_filters_in, nr_filters_out, filter_size, stride, output_padding=output_padding))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x, output_padding=None):
        if output_padding is None:
            x = self.deconv(x)
        else:
            x = F.conv_transpose2d(
                x,
                self.deconv.weight,
                self.deconv.bias,
                stride=self.stride,
                padding=self.deconv.padding,
                output_padding=output_padding,
                dilation=self.deconv.dilation,
                groups=self.deconv.groups,
            )
        xs = list(x.shape)
        return x[:, :, :(xs[2] - self.filter_size[0] + 1), :(xs[3] - self.filter_size[1] + 1)]


# IMPLEMENT PIXELCNN++ HERE
class PixelCNNpp(nn.Module):
    def __init__(self,
                 nr_resnet: int,
                 nr_filters: int,
                 nr_logistic_mix: int,
                 resnet_nonlinearity: str,
                 dropout_p: float,
                 input_channels: int = 1):

        super().__init__()
        if resnet_nonlinearity == "concat_elu" :
            self.resnet_nonlinearity = concat_elu
        elif resnet_nonlinearity == "elu" :
            self.resnet_nonlinearity = F.elu
        elif resnet_nonlinearity == "relu" :
            self.resnet_nonlinearity = F.relu
        else:
            raise Exception("Only concat elu, elu and relu are supported as resnet nonlinearity.")  # noqa: TRY002

        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.dropout_p = dropout_p
        self.input_channels = input_channels


        # UP PASS blocks
        self.u_init = DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2,3), stride=(1,1), shift_output_down=True)
        self.ul_init = nn.ModuleList([DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(1,3), stride=(1,1), shift_output_down=True),
                                      DownRightShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2,1), stride=(1,1), shift_output_right_down=True)])

        self.gated_resnet_block_u_up_1 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=0) for _ in range(nr_resnet)])
        self.gated_resnet_block_ul_up_1 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])

        self.downsize_u_1 = DownShiftedConv2d(nr_filters, nr_filters, filter_size=(2,3), stride=(2,2))
        self.downsize_ul_1 = DownRightShiftedConv2d(nr_filters, nr_filters, filter_size=(2,2), stride=(2,2))

        self.gated_resnet_block_u_up_2 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=0) for _ in range(nr_resnet)])
        self.gated_resnet_block_ul_up_2 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])

        self.downsize_u_2 = DownShiftedConv2d(nr_filters, nr_filters, filter_size=(2,3), stride=(2,2))
        self.downsize_ul_2 = DownRightShiftedConv2d(nr_filters, nr_filters, filter_size=(2,2), stride=(2,2))

        self.gated_resnet_block_u_up_3 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=0) for _ in range(nr_resnet)])
        self.gated_resnet_block_ul_up_3 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])


        # DOWN PASS blocks
        self.gated_resnet_block_u_down_1 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet)])
        self.gated_resnet_block_ul_down_1 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=2) for _ in range(nr_resnet)])

        self.upsize_u_1 = DownShiftedDeconv2d(nr_filters, nr_filters, filter_size=(2,3), stride=(2,2))
        self.upsize_ul_1 = DownRightShiftedDeconv2d(nr_filters, nr_filters, filter_size=(2,2), stride=(2,2))

        self.gated_resnet_block_u_down_2 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet + 1)])
        self.gated_resnet_block_ul_down_2 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=2) for _ in range(nr_resnet + 1)])

        self.upsize_u_2 = DownShiftedDeconv2d(nr_filters, nr_filters, filter_size=(2,3), stride=(2,2))
        self.upsize_ul_2 = DownRightShiftedDeconv2d(nr_filters, nr_filters, filter_size=(2,2), stride=(2,2))

        self.gated_resnet_block_u_down_3 = nn.ModuleList([GatedResnet(nr_filters, DownShiftedConv2d, self.resnet_nonlinearity, skip_connection=1) for _ in range(nr_resnet + 1)])
        self.gated_resnet_block_ul_down_3 = nn.ModuleList([GatedResnet(nr_filters, DownRightShiftedConv2d, self.resnet_nonlinearity, skip_connection=2) for _ in range(nr_resnet + 1)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)

    def forward(self, x: th.Tensor, c: th.Tensor) -> th.Tensor:
        xs = list(x.shape)
        padding = th.ones(xs[0], 1, xs[2], xs[3], device=x.device)
        x = th.cat((x, padding), dim=1)  # add extra channel for padding

        output_padding_list = []

        # UP PASS ("encoder")
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]

        for i in range(self.nr_resnet):
            u_list.append(self.gated_resnet_block_u_up_1[i](u_list[-1], a=None, h=c))
            ul_list.append(self.gated_resnet_block_ul_up_1[i](ul_list[-1], a=u_list[-1], h=c))

        u_list.append(self.downsize_u_1(u_list[-1]))
        ul_list.append(self.downsize_ul_1(ul_list[-1]))

        # Handle images with odd height/width
        # print(f"Before first downsize: u.shape[2]: {u_list[-1].shape[2]}, u.shape[3]: {u_list[-1].shape[3]}, u.shape[2]: {u_list[-2].shape[2]}, u.shape[3]: {u_list[-2].shape[3]}")
        pad_height = 1
        pad_width = 1
        if u_list[-2].shape[2] % u_list[-1].shape[2] != 0:
            pad_height = 0
        if u_list[-2].shape[3] % u_list[-1].shape[3] != 0:
            pad_width = 0
        output_padding_list.append((pad_height, pad_width))

        for i in range(self.nr_resnet):
            u_list.append(self.gated_resnet_block_u_up_2[i](u_list[-1], a=None, h=c))
            ul_list.append(self.gated_resnet_block_ul_up_2[i](ul_list[-1], a=u_list[-1], h=c))

        u_list.append(self.downsize_u_2(u_list[-1]))
        ul_list.append(self.downsize_ul_2(ul_list[-1]))

        # Handle images with odd height/width
        pad_height = 1
        pad_width = 1
        if u_list[-2].shape[2] % u_list[-1].shape[2] != 0:
            pad_height = 0
        if u_list[-2].shape[3] % u_list[-1].shape[3] != 0:
            pad_width = 0
        output_padding_list.append((pad_height, pad_width))

        for i in range(self.nr_resnet):
            u_list.append(self.gated_resnet_block_u_up_3[i](u_list[-1], a=None, h=c))
            ul_list.append(self.gated_resnet_block_ul_up_3[i](ul_list[-1], a=u_list[-1], h=c))

        # for i, u in enumerate(u_list):
        #     print(f"u_list[{i}] shape: {u.shape}")
        # for i, ul in enumerate(ul_list):
        #     print(f"ul_list[{i}] shape: {ul.shape}")
        # print(f"output_padding_list: {output_padding_list}")
        # DOWN PASS ("decoder")
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(self.nr_resnet):
            u = self.gated_resnet_block_u_down_1[i](u, a=u_list.pop(), h=c)
            ul = self.gated_resnet_block_ul_down_1[i](ul, a=th.cat((u, ul_list.pop()), dim=1), h=c)

        #print(f"After first down pass: u shape: {u.shape}, ul shape: {ul.shape}")
        u = self.upsize_u_1(u, output_padding=output_padding_list[-1])
        ul = self.upsize_ul_1(ul, output_padding=output_padding_list[-1])
        # print(f"After first upsize: u shape: {u.shape}, ul shape: {ul.shape}")

        for i in range(self.nr_resnet + 1):
            u = self.gated_resnet_block_u_down_2[i](u, a=u_list.pop(), h=c)
            ul = self.gated_resnet_block_ul_down_2[i](ul, a=th.cat((u, ul_list.pop()), dim=1), h=c)

        # print(f"After second down pass: u shape: {u.shape}, ul shape: {ul.shape}")
        u = self.upsize_u_2(u, output_padding=output_padding_list[-2])
        ul = self.upsize_ul_2(ul, output_padding=output_padding_list[-2])
        # print(f"After second upsize: u shape: {u.shape}, ul shape: {ul.shape}")

        for i in range(self.nr_resnet + 1):
            u = self.gated_resnet_block_u_down_3[i](u, a=u_list.pop(), h=c)
            ul = self.gated_resnet_block_ul_down_3[i](ul, a=th.cat((u, ul_list.pop()), dim=1), h=c)


        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == 0
        assert len(ul_list) == 0

        return x_out

def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation that prevents overflow."""
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = th.max(x, dim=axis)
    m2, _ = th.max(x, dim=axis, keepdim=True)
    return m + th.log(th.sum(th.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """Numerically stable log_softmax implementation that prevents overflow."""
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = th.max(x, dim=axis, keepdim=True)
    return x - m - th.log(th.sum(th.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
        """Log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval."""
        # Pytorch ordering
        x = x.permute(0, 2, 3, 1)
        l = l.permute(0, 2, 3, 1)
        xs = list(x.shape) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
        ls = list(l.shape) # predicted distribution, e.g. (B,32,32,100)
        # different for 3 channels: / 10
        nr_mix = int(ls[-1] / 3) # here and below: unpacking the params of the mixture of logistics
        logit_probs = l[:,:,:,:nr_mix]
        # different for 3 channels: nr_mix * 3 #, coeff
        l = l[:, :, :, nr_mix:].contiguous().view([*xs, nr_mix * 2]) # 3 for mean, scale
        means = l[:,:,:,:,:nr_mix]
        log_scales = th.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
        # for 3 channels:
        # coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
        x = x.contiguous()
        zeros = th.zeros([*xs, nr_mix], device=x.device)
        x = x.unsqueeze(-1) + zeros
        # for 3 channels:
        # m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
        #             * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

        # m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        #             coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
        # means = th.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

        centered_x = x - means
        inv_stdv = th.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1./255.)
        cdf_plus = F.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1./255.)
        cdf_min = F.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2.*F.softplus(mid_in)

        # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen here)

        # this is what is really done, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value

        # log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
        log_probs = th.where(x < -0.999, log_cdf_plus, th.where(x > 0.999, log_one_minus_cdf_min, th.where(cdf_delta > 1e-5, th.log(th.clamp(cdf_delta, min=1e-12)), log_pdf_mid - np.log(127.5))))  # noqa: PLR2004
        log_probs = th.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
        return -th.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = th.zeros((*tensor.size(), n), device=tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = list(l.shape)
    xs = [*ls[:-1], 1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view([*xs, nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = th.empty_like(logit_probs).uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.detach() - th.log(- th.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view([*xs[:-1], 1, nr_mix])
    # select logistic parameters
    means = th.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = th.clamp(th.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = th.empty_like(means).uniform_(1e-5, 1. - 1e-5)
    x = means + th.exp(log_scales) * (th.log(u) - th.log(1. - u))
    x0 = th.clamp(th.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    #print(f"Design shape: {design_shape}")
    conditions = problem.conditions_keys
    nr_conditions = len(conditions)


    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    #device = th.device("cpu")

    # Loss function
    loss_operator = discretized_mix_logistic_loss

    # Initialize model
    model = PixelCNNpp(
        nr_resnet=args.nr_resnet,
        nr_filters=args.nr_filters,
        nr_logistic_mix=args.nr_logistic_mix,
        resnet_nonlinearity=args.resnet_nonlinearity,
        dropout_p=args.dropout_p,
        input_channels=1
    )

    model.to(device)
    # loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]
    condition_tensors = [training_ds[key][:] for key in problem.conditions_keys]

    training_ds = th.utils.data.TensorDataset(training_ds["optimal_design"][:], *condition_tensors) # .flatten(1) ?

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimzer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2)) # add other args if necessary

    # @th.no_grad()
    # def sample_designs(model: PixelCNNpp, design_shape: tuple[int, int, int], n_designs: int = 25) -> tuple[th.Tensor, th.Tensor]:
    #     """Samples n_designs designs."""
    #     model.eval()
    #     data = torch.zeros(n_designs, design_shape[0], design_shape[1], design_shape[2])
    #     data = data.cuda()
    #     with torch.no_grad():
    #         for i in range(design_shape[1]):
    #             for j in range(design_shape[2]):
    #                 out = model(data, sample=True)
    #                 out_sample = sample_from_discretized_mix_logistic(out, args.nr_logistic_mix)
    #                 data[:, :, i, j] = out_sample.data[:, :, i, j]
    #     return data


    @th.no_grad()
    def sample_designs(model: PixelCNNpp, design_shape: tuple[int, int, int], dim: int = 1, n_designs: int = 25) -> tuple[th.Tensor, th.Tensor]:
        """Samples n_designs designs using dataset conditions."""
        model.eval()
        device = next(model.parameters()).device

        linspaces = [
            th.linspace(conds[:, i].min(), conds[:, i].max(), n_designs, device=device) for i in range(conds.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1).reshape(-1, nr_conditions, 1, 1)

        # # Build per-condition min/max from the dataset tensors (device-safe)
        # # `condition_tensors` is defined in the outer scope above when the
        # # dataset is prepared: it's a list of 1-D tensors (one per condition).
        # all_conditions = th.stack(condition_tensors, dim=1).to(device)  # [N_examples, nr_conditions]
        # conds_min = all_conditions.amin(dim=0)
        # conds_max = all_conditions.amax(dim=0)

        # # Create a sweep of condition vectors between min and max (diagonal sweep)
        # steps = th.linspace(0.0, 1.0, n_designs, device=device).unsqueeze(1)  # [n_designs, 1]
        # encoder_hidden_states = conds_min.unsqueeze(0) + steps * (conds_max - conds_min).unsqueeze(0)
        # # reshape to [B, nr_conditions, 1, 1] as expected by the model's conditional input
        # encoder_hidden_states = encoder_hidden_states.view(n_designs, len(problem.conditions_keys), 1, 1).to(device)

        # Prepare an empty image batch on the same device as the model
        data = th.zeros((n_designs, dim, *design_shape), device=device)

        # Autoregressive pixel sampling: iterate spatial positions and condition on
        # previously sampled pixels and the desired_conds.
        for i in range(design_shape[0]):
            for j in range(design_shape[1]):
                # print(f"Sampling pixel ({i}, {j})")
                out = model(data, desired_conds)
                # print(f"out shape: {out.shape}")
                out_sample = sample_from_discretized_mix_logistic(out, args.nr_logistic_mix)
                # print(f"out_sample shape: {out_sample.shape}")
                # `out_sample` has shape [B, 1, H, W]; copy the sampled value for (i,j)
                data[:, :, i, j] = out_sample.data[:, :, i, j] #out_sample[:, :, i, j] # out_sample.data[:, :, i, j]
            #print(f"Completed row {i+1}/{design_shape[0]}")

        #print(f"final data shape: {data.shape}")
        #print(f"desired_conds shape: {desired_conds.shape}")
        return data, desired_conds



    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        model.train()
        for i, data in enumerate(dataloader):
            designs = data[0].unsqueeze(dim=1) # add channel dim

            #print(designs.shape)
            #print(data[1:])

            conds = th.stack((data[1:]), dim=1).reshape(-1, nr_conditions, 1, 1)
            # print(f"conds.shape: {conds.shape}")
            #print(conds)


            # in PixelCNNpp.__init__
            # h_lin = nn.Linear(nr_conditions, 2 * args.nr_filters)

            # # in forward, when `h` is [B, nr_conditions, 1, 1]
            # h_flat = conds.view(conds.size(0), -1)                        # [B, nr_conditions]
            # h_proj = h_lin(h_flat).unsqueeze(-1).unsqueeze(-1)  # [B, 2*nr_filters, 1, 1]
            # print(h_proj.shape)
            # print(h_proj)

            # conds = th.stack((data[1:]), dim=1).reshape(-1, nr_conditions, 1, 1)
            # print(conds.shape)
            # print(conds)

            batch_start_time = time.time()
            out = model(designs, conds)
            # Compute loss
            loss = loss_operator(designs, out)
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            optimizer.step()


            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}]] [{time.time() - batch_start_time:.2f} sec]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs

                    designs, desired_conds = sample_designs(model, design_shape, dim=1, n_designs=5)
                    fig, axes = plt.subplots(1, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot the image created by each output
                    for j, tensor in enumerate(designs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])  # Extract x and y coordinates
                        #print(f"img shape: {img.shape}")
                        dc = desired_conds[j].cpu().squeeze() # Extract design conditions
                        #print(f"dc shape: {dc.shape}")
                        axes[j].imshow(img)  # image plot
                        title = [(problem.conditions_keys[i][0], f"{dc[i]:.2f}") for i in range(nr_conditions)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)  # Set title
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                #if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                if args.save_model and (((epoch + 1) % args.model_storage_interval == 0) or (epoch == args.n_epochs - 1)) and i == len(dataloader) - 1:
                    ckpt_model = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "model": model.state_dict(),
                        "optimizer_generator": optimizer.state_dict(),
                        "loss": loss.item(),
                    }

                    th.save(ckpt_model, f"model_epoch{epoch+1}.pth")
                    if args.track:
                        artifact_model = wandb.Artifact(f"{args.problem_id}_{args.algo}_model", type="model")
                        artifact_model.add_file(f"model_epoch{epoch+1}.pth")

                        wandb.log_artifact(artifact_model, aliases=[f"seed_{args.seed}"])

    wandb.finish()
