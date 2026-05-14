from torch import nn


def convert_str_to_activ(activ_str: str):
    """ Converts a string to a PyTorch activation function.
    @activ_str: string, the name of the activation function.
    @return: PyTorch activation function.
    """
    activ_str = activ_str.upper()

    if activ_str == 'RELU':
        return nn.ReLU()
    elif activ_str == 'SIGMOID':
        return nn.Sigmoid()
    elif activ_str == 'TANH':
        return nn.Tanh()
    elif activ_str == 'LEAKYRELU':
        return nn.LeakyReLU()
    elif activ_str == 'PRELU':
        return nn.PReLU()
    elif activ_str == 'SOFTMAX':
        return nn.Softmax(dim=-1)
    elif activ_str == 'ELU':
        return nn.ELU()
    elif activ_str == 'SELU':
        return nn.SELU()
    elif activ_str == 'CELU':
        return nn.CELU()
    elif activ_str == 'GLU':
        return nn.GLU()
    elif activ_str == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f"Activation function '{activ_str}' not supported.")
