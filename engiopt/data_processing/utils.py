import numpy as np
import torch


class scaler:
    def __init__(self, mean_std):
        self.mean = mean_std[0]
        self.std = mean_std[1]

    def transform(self, x):
        if isinstance(x, torch.Tensor):
            mean_temp = torch.tensor(self.mean, dtype=torch.float32).to(x.device)
            std_temp = torch.tensor(self.std, dtype=torch.float32).to(x.device)
            return (x - mean_temp) / std_temp
        else:
            return (x - self.mean) / self.std

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            mean_temp = torch.tensor(self.mean, dtype=torch.float32).to(x.device)
            std_temp = torch.tensor(self.std, dtype=torch.float32).to(x.device)
            return (x * std_temp) + mean_temp
        else:
            return (x * self.std) + self.mean