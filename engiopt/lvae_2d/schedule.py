import numpy as np
import torch
from copy import deepcopy

def polynomial_schedule(w, N, p=1, w_init=[1., 0.], M=0):
    w = torch.as_tensor(w, dtype=torch.float)
    w_init = torch.as_tensor(w_init, dtype=torch.float)
    def poly_w(epoch):
        if epoch >=N:
            return w
        if epoch < M:
            return w_init
        else:
            k = (epoch-M) ** p / ((N-M) ** p)
            w_n = w_init + (w - w_init) * k
            return w_n
    return poly_w
