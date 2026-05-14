import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_eps = 1e-7


class BezierLayer(nn.Module):
    r"""Produces data points on a Bezier curve."""

    def __init__(self, n_control_points: int, n_data_points: int) -> None:
        super().__init__()
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

    def forward(self, input: Tensor, control_points: Tensor, weights: Tensor):
        cp, w = self._check_consistency(control_points, weights)
        bs, pv, intvls = self.generate_bernstein_polynomial(input)
        dp = (cp * w) @ bs / (w @ bs + _eps)
        return dp, pv, intvls

    def _check_consistency(self, control_points: Tensor, weights: Tensor):
        assert control_points.shape[-1] == self.n_control_points, "The number of control points is not consistent."
        assert weights.shape[-1] == self.n_control_points, "The number of weights is not consistent."
        assert weights.shape[1] == 1, "There should be only one weight corresponding to each control point."
        return control_points, weights

    def generate_bernstein_polynomial(self, intvls: Tensor):
        pv = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1)
        pw1 = torch.arange(0.0, self.n_control_points, device=intvls.device).view(1, -1, 1)
        pw2 = torch.flip(pw1, (1,))
        lbs = (
            pw1 * torch.log(pv + _eps)
            + pw2 * torch.log(1 - pv + _eps)
            + torch.lgamma(torch.tensor(self.n_control_points, device=intvls.device) + _eps).view(1, -1, 1)
            - torch.lgamma(pw1 + 1 + _eps)
            - torch.lgamma(pw2 + 1 + _eps)
        )
        bs = torch.exp(lbs)
        return bs, pv, intvls

    def extra_repr(self) -> str:
        return f"n_control_points={self.n_control_points}, n_data_points={self.n_data_points}"


class BezierAutoencoder(nn.Module):
    r"""Bezier autoencoder with fixed first and last control points by default."""

    def __init__(
        self,
        n_control_points: int,
        n_data_points: int,
        cp_layers: list = [64, 32],
        w_layers: list = [64, 32],
        batch_size=1,
        auto_batch=False,
        w_multiplier=2,
        w_min=0.1,
        cpx_bound=[-0.75, 1.00],
        cpy_bound=[-0.75, 0.75],
        unrestricted=False,
        activation="tanh",
        activation_cp="gelu",
        activation_w="gelu",
        learn_weights=True,
        n_channels=2,
        no_activ=False,
        x_scaler=[0.0, 1.0],
        y_scaler=[0.0, 1.0],
    ):
        super().__init__()
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.cp_layers = cp_layers
        self.w_layers = w_layers
        self.auto_batch = auto_batch
        self.batch_size = batch_size
        self.unrestricted = unrestricted
        self.cpx_bound = cpx_bound
        self.cpy_bound = cpy_bound
        self.w_multiplier = w_multiplier
        self.w_min = w_min
        self.learn_weights = learn_weights

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        scale_x = x_scaler[1]
        shift_x = x_scaler[0]
        scale_y = y_scaler[1]
        shift_y = y_scaler[0]
        trailing_point = [scale_x * (1.0 - shift_x), scale_y * (0.0 - shift_y)]

        self.base_intvls_init = torch.ones((1, n_data_points), requires_grad=False) * 1.0 / (n_data_points - 1)
        self.base_intvls_init[0, 0] = 0
        self.register_buffer("base_intvls", self.base_intvls_init, persistent=False)
        self.register_buffer("base_intvls_batch", self.base_intvls_init, persistent=False)

        if not self.unrestricted:
            base_loop_point = torch.tensor(trailing_point, requires_grad=False).view(1, 2, 1)

            self.register_buffer(
                "cp_loop_point",
                base_loop_point,
                persistent=False,
            )
            self.register_buffer(
                "cp_loop_point_batch",
                base_loop_point.repeat(batch_size, 1, 1),
                persistent=False,
            )

        self.bezier_decoder = BezierLayer(n_control_points=n_control_points, n_data_points=n_data_points)

        self.activation = convert_str_to_activ(activation)
        self.activation_cp = convert_str_to_activ(activation_cp)
        self.activation_w = convert_str_to_activ(activation_w)

        if activation_cp == nn.PReLU():
            self.activation_cp = nn.PReLU(cp_layers[0])

        self.cp_encoder = nn.ModuleList([nn.Linear(n_channels * n_data_points, cp_layers[0]), self.activation_cp])
        for i in range(len(cp_layers) - 1):
            if activation_cp == nn.PReLU():
                self.activation_cp = nn.PReLU(cp_layers[i + 1])
            self.cp_encoder.extend([nn.Linear(cp_layers[i], cp_layers[i + 1]), self.activation_cp])

        if self.unrestricted:
            if no_activ:
                self.cp_encoder.extend([nn.Linear(cp_layers[-1], 2 * n_control_points)])
            else:
                self.cp_encoder.extend([nn.Linear(cp_layers[-1], 2 * n_control_points), nn.Tanh()])
        else:
            if no_activ:
                self.cp_encoder.extend([nn.Linear(cp_layers[-1], 2 * n_control_points - 4)])
            else:
                self.cp_encoder.extend([nn.Linear(cp_layers[-1], 2 * n_control_points - 4), nn.Tanh()])

        self.w_encoder = nn.ModuleList([nn.Linear(n_channels * n_data_points, w_layers[0]), self.activation_w])
        for i in range(len(w_layers) - 1):
            if activation_w == nn.PReLU():
                self.activation_w = nn.PReLU(w_layers[i + 1])
            self.w_encoder.extend([nn.Linear(w_layers[i], w_layers[i + 1]), self.activation_w])

        if self.unrestricted:
            self.w_encoder.extend([nn.Linear(w_layers[-1], n_control_points), nn.Sigmoid()])
            self.w_size = n_control_points
        else:
            self.w_encoder.extend([nn.Linear(w_layers[-1], n_control_points - 2), nn.Sigmoid()])
            self.w_size = n_control_points - 2

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.base_intvls_batch = self.base_intvls.repeat(self.batch_size, 1)
        if not self.unrestricted:
            self.cp_loop_point_batch = self.cp_loop_point.repeat(self.batch_size, 1, 1)

    def change_auto_batch(self, auto_batch):
        self.auto_batch = auto_batch

    def cp_transform(self, cp_ae):
        cp_ae = (cp_ae + 1) / 2
        cp_ae = cp_ae * torch.tensor(
            [self.cpx_bound[1] - self.cpx_bound[0], self.cpy_bound[1] - self.cpy_bound[0]],
            device=cp_ae.device,
        ).view(1, 2, 1)
        cp_ae = cp_ae + torch.tensor([self.cpx_bound[0], self.cpy_bound[0]], device=cp_ae.device).view(1, 2, 1)
        return cp_ae

    def forward(self, x):
        cp, w = self.encode(x)
        dp, pv, intvls = self.decode(cp, w)
        return dp, pv, intvls, cp, w

    def decode(self, cp, w):
        if self.auto_batch:
            intvl_b = self.base_intvls.repeat(cp.shape[0], 1).to(cp.device)
            dp, pv, intvls = self.bezier_decoder(intvl_b, cp, w)
        else:
            dp, pv, intvls = self.bezier_decoder(self.base_intvls_batch, cp, w)
        return dp, pv, intvls

    def decode_z(self, z, z_ae_mode=False, normalized_data=False, denormalize_output=False):
        w = z[:, 0, :].unsqueeze(1)
        cp = z[:, 1:, :]

        if normalized_data:
            cp = self.cp_transform(cp)
            w = ((w + 1) / 2) * (self.w_multiplier - self.w_min) + self.w_min

        if z_ae_mode and not self.unrestricted:
            if self.auto_batch:
                cp_loop_point_b = self.cp_loop_point.repeat(cp.shape[0], 1, 1).to(cp.device)
                cp = torch.cat((cp_loop_point_b, cp, cp_loop_point_b), dim=2)
            else:
                cp = torch.cat((self.cp_loop_point_batch, cp, self.cp_loop_point_batch), dim=2)
            w = nn.ConstantPad1d((1, 1), 1.0)(w)

        dp, pv, intvls = self.decode(cp, w)

        if denormalize_output:
            dp[:, 0, :] = dp[:, 0, :] / self.x_scaler[1] + self.x_scaler[0]
            dp[:, 1, :] = dp[:, 1, :] / self.y_scaler[1] + self.y_scaler[0]

        return dp, pv, intvls

    def encode(self, x, return_z=False, z_ae_mode=False, normalize_output=False, scale_input=False):
        if scale_input:
            x[:, 0, :] = (x[:, 0, :] - self.x_scaler[0]) * self.x_scaler[1]
            x[:, 1, :] = (x[:, 1, :] - self.y_scaler[0]) * self.y_scaler[1]

        x = x.flatten(start_dim=1)

        if self.learn_weights:
            w = x
        else:
            w = torch.ones(x.shape[0], 1, self.w_size, device=x.device)

        cp = x

        for layer in self.cp_encoder:
            cp = layer(cp)

        if self.unrestricted:
            cp = cp.view(-1, 2, self.n_control_points)
        else:
            cp = cp.view(-1, 2, self.n_control_points - 2)

        if return_z and normalize_output:
            cp_out = cp
            if not self.unrestricted:
                if self.auto_batch:
                    cp_loop_point_b = self.cp_loop_point.repeat(cp_out.shape[0], 1, 1)
                    cp_out = torch.cat((cp_loop_point_b, cp_out, cp_loop_point_b), dim=2)
                else:
                    cp_out = torch.cat((self.cp_loop_point_batch, cp_out, self.cp_loop_point_batch), dim=2)

        if z_ae_mode:
            if normalize_output:
                cp_ae = cp
                cp = self.cp_transform(cp)
            else:
                cp = self.cp_transform(cp)
                cp_ae = cp

        if not self.unrestricted:
            if self.auto_batch:
                cp_loop_point_b = self.cp_loop_point.repeat(cp.shape[0], 1, 1)
                cp = torch.cat((cp_loop_point_b, cp, cp_loop_point_b), dim=2)
            else:
                cp = torch.cat((self.cp_loop_point_batch, cp, self.cp_loop_point_batch), dim=2)

        if self.learn_weights:
            for layer in self.w_encoder:
                w = layer(w)

        w = w * self.w_multiplier

        if self.unrestricted:
            w = w.view(-1, 1, self.n_control_points)
        else:
            w = w.view(-1, 1, self.n_control_points - 2)

        w = torch.clamp(w, self.w_min, self.w_multiplier)

        if z_ae_mode:
            if normalize_output:
                w_ae = 2 * (w - self.w_min) / (self.w_multiplier - self.w_min) - 1
            else:
                w_ae = w

        if not self.unrestricted:
            w = nn.ConstantPad1d((1, 1), 1.0)(w)

        if z_ae_mode:
            z_ae = torch.cat([w_ae, cp_ae], dim=1)
            return z_ae

        elif return_z:
            if normalize_output:
                w_out = 2 * (w - self.w_min) / (self.w_multiplier - self.w_min) - 1
                z = torch.cat([w_out, cp_out], dim=1)
            else:
                z = torch.cat([w, cp], dim=1)
            return z

        else:
            return cp, w


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def loss_reg_fn(y, x, cp, w, reg_fac=0.001):
    mean_reg = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
    return F.mse_loss(y, x) + reg_fac * mean_reg


class GaussianFourierFeatureTransform(torch.nn.Module):
    def __init__(self, num_input_channels, mapping_size=8, scale=10):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        batches, channels, width = x.shape

        assert channels == self._num_input_channels, (
            f"Expected input to have {self._num_input_channels} channels "
            f"(got {channels} channels)"
        )

        x = x.permute(0, 2, 1).reshape(batches * width, channels)
        x = x @ self._B.to(x.device)
        x = x.view(batches, width, self._mapping_size)
        x = x.permute(0, 2, 1)
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


def convert_str_to_activ(activ_str: str):
    activ_str = activ_str.upper()

    if activ_str == "RELU":
        return nn.ReLU()
    elif activ_str == "SIGMOID":
        return nn.Sigmoid()
    elif activ_str == "TANH":
        return nn.Tanh()
    elif activ_str == "LEAKYRELU":
        return nn.LeakyReLU()
    elif activ_str == "PRELU":
        return nn.PReLU()
    elif activ_str == "SOFTMAX":
        return nn.Softmax(dim=-1)
    elif activ_str == "ELU":
        return nn.ELU()
    elif activ_str == "SELU":
        return nn.SELU()
    elif activ_str == "CELU":
        return nn.CELU()
    elif activ_str == "GLU":
        return nn.GLU()
    elif activ_str == "GELU":
        return nn.GELU()
    else:
        raise ValueError(f"Activation function '{activ_str}' not supported.")