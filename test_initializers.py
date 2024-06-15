import pytest
import numpy as np
import torch
from scipy.stats import kstest

from tensor import Tensor

# Increasing sample size to reduce variability
SAMPLE_SIZE = 300000

@pytest.mark.parametrize("loc, scale, size", [
    (0.0, 1.0, (SAMPLE_SIZE,)),
    (1.0, 2.0, (SAMPLE_SIZE,)),
    (-1.0, 0.5, (SAMPLE_SIZE,))
])
def test_normal_initializer(loc, scale, size):
    my_tensor = Tensor.normal(shape=size, mean=loc, std=scale).data
    torch_tensor = torch.empty(size)
    torch.nn.init.normal_(torch_tensor, mean=loc, std=scale)
    torch_tensor = torch_tensor.numpy()

    my_mean, my_std = np.mean(my_tensor), np.std(my_tensor)
    torch_mean, torch_std = np.mean(torch_tensor), np.std(torch_tensor)

    assert np.isclose(my_mean, torch_mean, atol=1e-2), f"Means differ: {my_mean} vs {torch_mean}"
    assert np.isclose(my_std, torch_std, atol=1e-2), f"Stds differ: {my_std} vs {torch_std}"

    ks_statistic, p_value = kstest(my_tensor, torch_tensor)
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape, a, b", [
    ((SAMPLE_SIZE,), 0.0, 1.0),
    ((SAMPLE_SIZE,), -1.0, 1.0),
    ((SAMPLE_SIZE,), 1.0, 2.0)
])
def test_uniform_initializer(shape, a, b):
    my_tensor = Tensor.uniform(shape, a=a, b=b).data
    torch_tensor = torch.empty(shape)
    torch.nn.init.uniform_(torch_tensor, a=a, b=b)
    torch_tensor = torch_tensor.numpy()

    my_mean, my_std = np.mean(my_tensor), np.std(my_tensor)
    torch_mean, torch_std = np.mean(torch_tensor), np.std(torch_tensor)

    assert np.isclose(my_mean, torch_mean, atol=1e-2), f"Means differ: {my_mean} vs {torch_mean}"
    assert np.isclose(my_std, torch_std, atol=1e-2), f"Stds differ: {my_std} vs {torch_std}"

    ks_statistic, p_value = kstest(my_tensor, torch_tensor)
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape, gain, mode", [
    ((SAMPLE_SIZE, 10), np.sqrt(5), 'fan_in'),
    ((SAMPLE_SIZE, 10), 1.0, 'fan_out'),
])
def test_kaiming_uniform_initializer(shape, gain, mode):
    my_tensor = Tensor.kaiming_uniform(shape, gain=gain, mode=mode).data
    torch_tensor = torch.empty(shape)
    torch.nn.init.kaiming_uniform_(torch_tensor, a=gain, mode=mode, nonlinearity='leaky_relu')
    torch_tensor = torch_tensor.numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape, gain, mode", [
    ((SAMPLE_SIZE, 10), np.sqrt(5), 'fan_in'),
    ((SAMPLE_SIZE, 10), 1.0, 'fan_out'),
])
def test_kaiming_normal_initializer(shape, gain, mode):
    my_tensor = Tensor.kaiming_normal(shape, gain=gain, mode=mode).data
    torch_tensor = torch.empty(shape)
    torch.nn.init.kaiming_normal_(torch_tensor, a=gain, mode=mode, nonlinearity='leaky_relu')
    torch_tensor = torch_tensor.numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape, gain", [
    ((SAMPLE_SIZE, 10), 1.0),
    ((SAMPLE_SIZE, 10), np.sqrt(2.0))
])
def test_xavier_uniform_initializer(shape, gain):
    my_tensor = Tensor.xavier_uniform(shape, gain=gain).data
    torch_tensor = torch.empty(shape)
    torch.nn.init.xavier_uniform_(torch_tensor, gain=gain)
    torch_tensor = torch_tensor.numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape, gain", [
    ((SAMPLE_SIZE, 10), 1.0),
    ((SAMPLE_SIZE, 10), np.sqrt(2.0))
])
def test_xavier_normal_initializer(shape, gain):
    my_tensor = Tensor.xavier_normal(shape, gain=gain).data
    torch_tensor = torch.empty(shape)
    torch.nn.init.xavier_normal_(torch_tensor, gain=gain)
    torch_tensor = torch_tensor.numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape", [
    (10, 10),
])
def test_zeros_initializer(shape):
    my_tensor = Tensor.zeros(shape).data
    torch_tensor = torch.zeros(shape).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("shape", [
    (10, 10),
])
def test_ones_initializer(shape):
    my_tensor = Tensor.ones(shape).data
    torch_tensor = torch.ones(shape).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("shape", [
    (10, 10),
])
def test_eye_initializer(shape):
    my_tensor = Tensor.eye(shape[0], shape[1]).data
    torch_tensor = torch.eye(shape[0], shape[1]).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("shape", [
    (SAMPLE_SIZE, 10),
])
def test_rand_initializer(shape):
    my_tensor = Tensor.rand(shape).data
    torch_tensor = torch.rand(shape).numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("shape", [
    (SAMPLE_SIZE, 10),
])
def test_randn_initializer(shape):
    print(shape)
    my_tensor = Tensor.randn(shape).data
    torch_tensor = torch.randn(shape).numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("low, high, size", [
    (0, 10, (SAMPLE_SIZE,)),
    (-5, 5, (SAMPLE_SIZE,)),
])
def test_randint_initializer(low, high, size):
    my_tensor = Tensor.randint(low, high, size).data
    torch_tensor = torch.randint(low, high, size).numpy()

    ks_statistic, p_value = kstest(my_tensor.flatten(), torch_tensor.flatten())
    assert p_value > 0.05, f"Distributions differ: KS statistic {ks_statistic}, p-value {p_value}"

@pytest.mark.parametrize("start, stop, step", [
    (0, 10, 1),
    (10, 20, 2),
])
def test_arange_initializer(start, stop, step):
    my_tensor = Tensor.arange(start, stop, step).data
    torch_tensor = torch.arange(start, stop, step).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("start, stop, num", [
    (0.0, 1.0, 50),
    (-1.0, 1.0, 100),
])
def test_linspace_initializer(start, stop, num):
    my_tensor = Tensor.linspace(start, stop, num).data
    torch_tensor = torch.linspace(start, stop, num).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("start, stop, num, base", [
    (0.0, 2.0, 50, 10.0),
    (1.0, 3.0, 100, 2.0),
])
def test_logspace_initializer(start, stop, num, base):
    my_tensor = Tensor.logspace(start, stop, num, base=base).data
    torch_tensor = torch.logspace(start, stop, num, base=base).numpy()

    assert np.allclose(my_tensor, torch_tensor)

@pytest.mark.parametrize("shape, fill_value", [
    ((10, 10), 5),
    ((5, 5), -3),
])
def test_full_initializer(shape, fill_value):
    my_tensor = Tensor.full(shape, fill_value).data
    torch_tensor = torch.full(shape, fill_value).numpy()

    assert np.allclose(my_tensor, torch_tensor)
