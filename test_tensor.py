import pytest
import torch
import numpy as np
from engine import Tensor  # Replace with the actual import

EPSILON = 1e-6  # Tolerance for floating point comparison

def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON)

def assert_gradients_equal(tensor1, tensor2):
    np.testing.assert_allclose(tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON)

def test_broadcast_addition():
    x = np.random.randn(2, 3)
    y = np.random.randn(3)  # This will be broadcast to (2, 3)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()  # Call sum() to get a scalar for backward()

    print("t1.grad:", t1.grad)
    print("x_torch.grad:", x_torch.grad)
    print("t2.grad:", t2.grad)
    print("y_torch.grad:", y_torch.grad)

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)

def test_add_debug():
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()

    print("t1.grad:", t1.grad)
    print("x_torch.grad:", x_torch.grad)
    print("t2.grad:", t2.grad)
    print("y_torch.grad:", y_torch.grad)

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)

def test_add():
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)

def test_matmul():
    x = np.random.randn(2, 3)
    y = np.random.randn(3, 4)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 @ t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch @ y_torch
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)

def test_mul():
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 * t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch * y_torch
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)

def test_relu():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.relu()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.relu(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_tanh():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.tanh()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.tanh(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_sigmoid():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.sigmoid()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.sigmoid(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_exp():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.exp()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.exp(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_log():
    x = np.random.randn(2, 3) + 1.1  # Avoid log(0)
    t1 = Tensor(x, requires_grad=True)
    result = t1.log()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.log(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_sum():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.sum()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.sum()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_mean():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.mean()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.mean()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_max():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.max()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.max()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_min():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.min()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.min()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_reshape():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.reshape((3, 2))
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.reshape((3, 2))
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

def test_pad():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = t1.pad(((1, 1), (2, 2)))
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.nn.functional.pad(x_torch, (2, 2, 1, 1))
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)

if __name__ == "__main__":
    pytest.main()
