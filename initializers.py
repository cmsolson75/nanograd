import numpy as np
import math

import functools
import operator

def calculate_fan_in(tensor):
    if len(tensor.shape) == 1:
        return tensor.shape[0]
    else:
        return tensor.shape[1]
    
def calculate_fan_in_and_fan_out(tensor):
    # adapted from pytorch
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    
    # Use functools.reduce to compute the product of the receptive field dimensions
    receptive_field_size = functools.reduce(operator.mul, tensor.shape[2:], 1) if dimensions > 2 else 1
    
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out
    

def calculate_gain(nonlinearity, param=None):
    """
    Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math: 1
    Conv{1,2,3}D      :math: 1
    Sigmoid           :math: 1
    Tanh              :math: 5 / 3
    ReLU              :math: sqrt{2}
    Leaky Relu        :math: sqrt(2 / (1 + negative_slope**2))
    SELU              :math: 3 / 4
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function
        param: optional parameter for the non-linear function

    References:
        Implementation adapted from:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py
        Copy pasta
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return (
            3.0 / 4
        )  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def kaiming_uniform(shape, a: float = 0, mode: float ='fan_in', nonlinearity: float ='leaky_relu'):
    tensor = np.empty(shape)
    # fan_in = calculate_fan_in(tensor) if mode == 'fan_in' else tensor.shape[0]
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    # bound = np.sqrt(3.0) * np.sqrt(2.0 / (1 + gain**2)) / np.sqrt(fan_in)
    tensor[:] = np.random.uniform(-bound, bound, size=tensor.shape)
    return tensor

def intercept_uniform(shape, weight):
    if not isinstance(weight, np.ndarray):
        raise ValueError("weight must be an instance of np.ndarray")
    
    tensor = np.empty(shape)
    fan_in, _ = calculate_fan_in_and_fan_out(weight)
    
    if fan_in <= 0:
        raise ValueError("fan_in must be greater than 0 to calculate the bound")
    
    bound = 1 / math.sqrt(fan_in)
    return np.random.uniform(-bound, bound, size=tensor.shape)

def kaiming_normal(shape, gain=math.sqrt(2), mode='fan_in'):
    tensor = np.empty(shape)
    fan_in = calculate_fan_in(tensor) if mode == 'fan_in' else tensor.shape[0]
    std = np.sqrt(2.0 / (1 + gain**2)) / np.sqrt(fan_in)
    tensor[:] = np.random.normal(0, std, size=tensor.shape)
    return tensor

def xavier_uniform(shape, gain=1.0):
    tensor = np.empty(shape)
    fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std
    tensor[:] = np.random.uniform(-bound, bound, size=tensor.shape)
    return tensor

def xavier_normal(shape, gain=1.0):
    tensor = np.empty(shape)
    fan_in, fan_out = tensor.shape[0], tensor.shape[1]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor[:] = np.random.normal(0, std, size=tensor.shape)
    return tensor

def zeros(shape):
    return np.zeros(shape)

def ones(shape):
    return np.ones(shape)

# Match torch api for this
def eye(N, M=None, k=0):
    M = N if M is None else M
    return np.eye(N, M, k)

def uniform(shape, a=0.0, b=1.0):
    tensor = np.empty(shape)
    tensor[:] = np.random.uniform(a, b, size=tensor.shape)
    return tensor

# this is my old interface: make all have this interface from old
def normal(loc, scale=1.0, size=None):
    tensor = np.random.normal(loc=loc, scale=scale, size=size)
    return tensor

def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        stop = start
        start = 0
    return np.arange(start, stop, step, dtype=dtype)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    return np.linspace(start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype)

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):
    return np.logspace(start, stop, num=num, endpoint=endpoint, base=base, dtype=dtype)

def full(shape, fill_value, dtype=None):
    return np.full(shape, fill_value, dtype=dtype)

def rand(shape):
    return np.random.rand(*shape)

def randn(shape):
    return np.random.randn(*shape)

def randint(low, high=None, size=None):
    return np.random.randint(low, high, size)
