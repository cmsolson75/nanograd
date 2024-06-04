import pytest
import torch
import numpy as np
from tensor_refactor import Tensor

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
    result.sum().backward()
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
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)



@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(3, 4)),
        (np.random.randn(3, 4), np.random.randn(4, 5)),
        (np.random.randn(5, 6), np.random.randn(6, 7)),
        (np.random.randn(2, 2), np.random.randn(2, 2)),  # Square matrices
        (np.random.randn(4, 2), np.random.randn(2, 1)),  # Column vector
        (np.random.randn(1, 3), np.random.randn(3, 4)),  # Row vector
    ],
)
def test_matmul_forward(x, y):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 @ t2

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch @ y_torch
    assert_tensors_equal(result, result_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(3, 4)),
        (np.random.randn(3, 4), np.random.randn(4, 5)),
        (np.random.randn(5, 6), np.random.randn(6, 7)),
        (np.random.randn(2, 2), np.random.randn(2, 2)),  # Square matrices
        (np.random.randn(4, 2), np.random.randn(2, 1)),  # Column vector
        (np.random.randn(1, 3), np.random.randn(3, 4)),  # Row vector
    ],
)
def test_matmul_backward(x, y, capsys):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano @ y_nano
    result.sum().backward()
    captured = capsys.readouterr()
    
    # Print the captured output for debugging purposes
    print("Captured stdout:", captured.out)
    print(x_nano.grad)
    print(y_nano.grad)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch @ y_torch
    result_torch.sum().backward()
    print(x_torch.grad)

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)


# Reduction op tests

@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(8, 1), 0, False),
        (np.random.randn(2, 3), 1, True),
        (np.random.randn(3), 0, False),
    ],
)
def test_sum_forward(x, axis, keepdims):
    t = Tensor(x, requires_grad=True)
    result = t.sum(axis=axis, keepdims=keepdims)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.sum(axis=axis, keepdim=keepdims)
    assert_tensors_equal(result, result_torch)


@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(8, 1), 0, False),
        (np.random.randn(3), 0, False),
    ],
)
def test_sum_backward(x, axis, keepdims, capsys):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.sum(axis=axis, keepdims=keepdims)
    result.sum().backward()
    print(result.shape)
    captured = capsys.readouterr()
    
    # Print the captured output for debugging purposes
    print("Captured stdout:", captured.out)
    print(x_nano.grad)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.sum(axis=axis, keepdim=keepdims)
    print(result_torch.size())
    result_torch.backward()
    print(x_torch.grad)

    assert_gradients_equal(x_nano, x_torch)



@pytest.mark.parametrize(
    "x, exponent",
    [
        (np.random.randn(2, 3), 2),
        (np.random.randn(3, 4), 3),
        (np.random.randn(8, 1), 0.5),
        (np.random.randn(2, 3), -1),
        (np.random.randn(3), 4),
    ],
)
def test_pow_forward(x, exponent):
    t = Tensor(x, requires_grad=True)
    result = t ** exponent

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch ** exponent
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, exponent",
    [
        (np.random.randn(2, 3), 2),
        (np.random.randn(3, 4), 3),
        (np.random.randn(8, 1), 0.5),
        (np.random.randn(2, 3), -1),
        (np.random.randn(3), 4),
    ],
)
def test_pow_backward(x, exponent, capsys):
    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch ** exponent
    print(result_torch)
    result_torch.sum().backward()

    x_nano = Tensor(x, requires_grad=True)
    result = x_nano ** exponent
    print(result)
    result.sum().backward()
    captured = capsys.readouterr()
    
    # Print the captured output for debugging purposes
    print("Captured stdout:", captured.out)

    assert_gradients_equal(x_nano, x_torch)



def test_basic_graph():
    # Initialize input and target
    x = np.random.randn(5, 2)  # Batch size of 5, input dimension of 2
    y = np.random.randn(5, 3)  # Target dimension of 3

    # Initialize weights and biases
    w1 = np.random.randn(2, 3)
    b1 = np.random.randn(1, 3)
    w2 = np.random.randn(3, 4)
    b2 = np.random.randn(1, 4)
    w3 = np.random.randn(4, 3)
    b3 = np.random.randn(1, 3)

    # Convert to Tensor
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    w1_nano = Tensor(w1, requires_grad=True)
    b1_nano = Tensor(b1, requires_grad=True)
    w2_nano = Tensor(w2, requires_grad=True)
    b2_nano = Tensor(b2, requires_grad=True)
    w3_nano = Tensor(w3, requires_grad=True)
    b3_nano = Tensor(b3, requires_grad=True)

    # Forward pass
    z1 = (x_nano @ w1_nano + b1_nano).relu()
    z2 = (z1 @ w2_nano + b2_nano).relu()
    y_hat_nano = z2 @ w3_nano + b3_nano

    # Loss calculation
    loss_nano = ((y_hat_nano - y_nano)**2).mean()
    loss_nano.backward()

    # PyTorch counterpart
    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    w1_torch = torch.tensor(w1, requires_grad=True)
    b1_torch = torch.tensor(b1, requires_grad=True)
    w2_torch = torch.tensor(w2, requires_grad=True)
    b2_torch = torch.tensor(b2, requires_grad=True)
    w3_torch = torch.tensor(w3, requires_grad=True)
    b3_torch = torch.tensor(b3, requires_grad=True)

    # Forward pass
    z1_torch = (x_torch @ w1_torch + b1_torch).relu()
    z2_torch = (z1_torch @ w2_torch + b2_torch).relu()
    y_hat_torch = z2_torch @ w3_torch + b3_torch

    # Loss calculation
    loss_torch = ((y_hat_torch - y_torch)**2).mean()
    loss_torch.backward()

    # Assertions to check if values and gradients match
    assert_tensors_equal(loss_nano, loss_torch)
    assert_gradients_equal(w1_nano, w1_torch)
    assert_gradients_equal(b1_nano, b1_torch)
    assert_gradients_equal(w2_nano, w2_torch)
    assert_gradients_equal(b2_nano, b2_torch)
    assert_gradients_equal(w3_nano, w3_torch)
    assert_gradients_equal(b3_nano, b3_torch)



@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_relu_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.relu()
    result_torch = torch.relu(x_torch)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_relu_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.relu()
    result_torch = torch.relu(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_tanh_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.tanh()
    result_torch = torch.tanh(x_torch)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_tanh_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    grad = np.random.randn(*x.shape)

    result_nano = x_nano.tanh()
    result_torch = torch.tanh(x_torch)

    result_nano.backward(Tensor(grad))
    result_torch.backward(torch.tensor(grad))

    assert_gradients_equal(x_nano, x_torch)


@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_sigmoid_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.sigmoid()
    result_torch = torch.sigmoid(x_torch)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3),
        np.random.randn(3, 4),
        np.random.randn(5, 1),
        np.random.randn(1, 5),
        np.random.randn(6, 6),
    ],
)
def test_sigmoid_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    grad = np.random.randn(*x.shape)

    result_nano = x_nano.sigmoid()
    result_torch = torch.sigmoid(x_torch)

    result_nano.backward(Tensor(grad))
    result_torch.backward(torch.tensor(grad))

    assert_gradients_equal(x_nano, x_torch)
