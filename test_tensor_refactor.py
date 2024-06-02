import pytest
import torch
import numpy as np
from tensor_refactor import Tensor  # Replace with the actual import

EPSILON = 1e-6  # Tolerance for floating point comparison


def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


def assert_gradients_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )

@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),  # Ensure compatible shapes
        (np.random.randn(8, 1), np.random.randn(8, 4)),
        (np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_add_forward(x, y):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    assert_tensors_equal(result, result_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(8, 1), np.random.randn(8, 4)),
        (np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_add_backward(x, y, capsys):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano + y_nano
    result.backward()
    captured = capsys.readouterr()
    
    # Print the captured output for debugging purposes
    print("Captured stdout:", captured.out)
    print(x_nano.grad)
    print(y_nano.grad)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()
    print(x_torch.grad)

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(8, 1), np.random.randn(8, 4)),
        (np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_mul_forward(x, y):
    t1 = Tensor(x)
    t2 = Tensor(y)
    result = t1 * t2

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)
    result_torch = x_torch * y_torch

    assert_tensors_equal(result, result_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(8, 1), np.random.randn(8, 4)),
        (np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_mul_backward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano + y_nano
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)