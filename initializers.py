import numpy as np
import math

def calculate_fan_in(tensor):
    if len(tensor.shape) == 1:
        return tensor.shape[0]
    else:
        return tensor.shape[1]

def kaiming_uniform(shape, gain=math.sqrt(2), mode='fan_in'):
    tensor = np.empty(shape)
    fan_in = calculate_fan_in(tensor) if mode == 'fan_in' else tensor.shape[0]
    bound = np.sqrt(3.0) * np.sqrt(2.0 / (1 + gain**2)) / np.sqrt(fan_in)
    tensor[:] = np.random.uniform(-bound, bound, size=tensor.shape)
    return tensor

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
# refactor: this is wrong

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
