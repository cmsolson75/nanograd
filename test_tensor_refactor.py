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
def test_abs_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.abs()
    result_torch = torch.abs(x_torch)

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
def test_abs_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.abs()
    result_torch = torch.abs(x_torch)

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
def test_neg_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.neg()
    result_torch = -x_torch

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
def test_neg_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.neg()
    result_torch = -x_torch

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


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
def test_mean_forward(x, axis, keepdims):
    t = Tensor(x, requires_grad=True)
    result = t.mean(axis=axis, keepdims=keepdims)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.mean(dim=axis, keepdim=keepdims)
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
def test_mean_backward(x, axis, keepdims, capsys):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.mean(axis=axis, keepdims=keepdims)
    result.sum().backward()
    captured = capsys.readouterr()

    # Print the captured output for debugging purposes
    print("Captured stdout:", captured.out)
    print(x_nano.grad)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.mean(dim=axis, keepdim=keepdims)
    result_torch.backward()
    print(x_torch.grad)

    assert_gradients_equal(x_nano, x_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,  # Adding 1 to avoid division by zero
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.rand(6, 6) + 1,
    ],
)
def test_reciprocal_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.reciprocal()
    result_torch = x_torch.reciprocal()

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.rand(6, 6) + 1,
    ],
)
def test_reciprocal_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.reciprocal()
    result_torch = x_torch.reciprocal()

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
def test_exp_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.exp()
    result_torch = x_torch.exp()

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
def test_exp_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.exp()
    result_torch = x_torch.exp()



    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.rand(6, 6) + 1,
    ],
)
def test_log_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.log()
    result_torch = x_torch.log()

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
def test_sin_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.sin()
    result_torch = torch.sin(x_torch)

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
def test_sin_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.sin()
    result_torch = torch.sin(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Tests for Max function
@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(5, 1), 0, False),
        (np.random.randn(1, 5), 0, True),
        (np.random.randn(6, 6), 1, False),
    ],
)
def test_max_forward(x, axis, keepdims):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.max(axis=axis, keepdims=keepdims)
    if axis is not None:
        result_torch = x_torch.max(dim=axis, keepdim=keepdims).values
    else:
        result_torch = x_torch.max()

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(5, 1), 0, False),
        (np.random.randn(1, 5), 0, True),
        (np.random.randn(6, 6), 1, False),
    ],
)
def test_max_backward(x, axis, keepdims):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.max(axis=axis, keepdims=keepdims)
    if axis is not None:
        result_torch = x_torch.max(dim=axis, keepdim=keepdims).values
    else:
        result_torch = x_torch.max()

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Tests for Min function
@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(5, 1), 0, False),
        (np.random.randn(1, 5), 0, True),
        (np.random.randn(6, 6), 1, False),
    ],
)
def test_min_forward(x, axis, keepdims):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    print(x_nano)
    result_nano = x_nano.min(axis=axis, keepdims=keepdims)
    print(result_nano)

    if axis is not None:
        result_torch = x_torch.min(dim=axis, keepdims=keepdims).values
    else:
        result_torch = x_torch.min()

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(3, 4), None, True),
        (np.random.randn(5, 1), 0, False),
        (np.random.randn(1, 5), 0, True),
        (np.random.randn(6, 6), 1, False),
    ],
)
def test_min_backward(x, axis, keepdims):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.min(axis=axis, keepdims=keepdims)

    if axis is not None:
        result_torch = x_torch.min(dim=axis, keepdim=keepdims).values
    else:
        result_torch = x_torch.min()

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(5, 1), np.random.randn(5, 1)),
        (np.random.randn(1, 5), np.random.randn(1, 5)),
        (np.random.randn(6, 6), np.random.randn(6, 6)),
    ],
)
def test_maximum_forward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result_nano = x_nano.maximum(y_nano)
    print(result_nano)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.maximum(x_torch, y_torch)
    print(result_torch)

    assert_tensors_equal(result_nano, result_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(5, 1), np.random.randn(5, 1)),
        (np.random.randn(1, 5), np.random.randn(1, 5)),
        (np.random.randn(6, 6), np.random.randn(6, 6)),
    ],
)
def test_maximum_backward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result_nano = x_nano.maximum(y_nano)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.maximum(x_torch, y_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)

@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(5, 1), np.random.randn(5, 1)),
        (np.random.randn(1, 5), np.random.randn(1, 5)),
        (np.random.randn(6, 6), np.random.randn(6, 6)),
    ],
)
def test_minimum_forward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result_nano = x_nano.minimum(y_nano)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.minimum(x_torch, y_torch)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(5, 1), np.random.randn(5, 1)),
        (np.random.randn(1, 5), np.random.randn(1, 5)),
        (np.random.randn(6, 6), np.random.randn(6, 6)),
    ],
)
def test_minimum_backward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result_nano = x_nano.minimum(y_nano)

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.minimum(x_torch, y_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)

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
def test_round_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.round()
    result_torch = torch.round(x_torch)

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
def test_round_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.round()
    result_torch = torch.round(x_torch)

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
def test_ceil_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.ceil()
    result_torch = torch.ceil(x_torch)

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
def test_ceil_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.ceil()
    result_torch = torch.ceil(x_torch)

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
def test_floor_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.floor()
    result_torch = torch.floor(x_torch)

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
def test_floor_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.floor()
    print(result_nano)
    result_torch = torch.floor(x_torch)
    print(result_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()
    print(x_nano.grad)
    print(x_torch.grad.detach().numpy())

    assert_gradients_equal(x_nano, x_torch)


@pytest.mark.parametrize(
    "x, shape",
    [
        (np.random.randn(2, 3), (3, 2)),
        (np.random.randn(3, 4), (4, 3)),
        (np.random.randn(8, 1), (4, 2)),
        (np.random.randn(2, 3), (6, )),
        (np.random.randn(3), (1, 3)),
    ],
)
def test_reshape_forward(x, shape):
    t1 = Tensor(x, requires_grad=True)
    result = t1.reshape(*shape)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.reshape(*shape)
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, shape",
    [
        (np.random.randn(2, 3), (3, 2)),
        (np.random.randn(3, 4), (4, 3)),
        (np.random.randn(8, 1), (4, 2)),
        (np.random.randn(2, 3), (6, )),
        (np.random.randn(3), (1, 3)),
    ],
)
def test_reshape_backward(x, shape):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.reshape(*shape)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.reshape(*shape)
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for Transpose
@pytest.mark.parametrize(
    "x, axes",
    [
        (np.random.randn(2, 3), None),
        (np.random.randn(3, 4), (1, 0)),
        (np.random.randn(8, 1), (1, 0)),
        (np.random.randn(2, 3, 4), (0, 2, 1)),
        (np.random.randn(3, 4, 5), (2, 1, 0)),
    ],
)
def test_transpose_forward(x, axes):
    t1 = Tensor(x, requires_grad=True)
    result = t1.transpose(axes)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.permute(axes) if axes else x_torch.T
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, axes",
    [
        (np.random.randn(2, 3), None),
        (np.random.randn(3, 4), (1, 0)),
        (np.random.randn(8, 1), (1, 0)),
        (np.random.randn(2, 3, 4), (0, 2, 1)),
        (np.random.randn(3, 4, 5), (2, 1, 0)),
    ],
)
def test_transpose_backward(x, axes):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.transpose(axes)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.permute(axes) if axes else x_torch.T
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for GetItem
@pytest.mark.parametrize(
    "x, index",
    [
        (np.random.randn(2, 3), (slice(0, 1),)),
        (np.random.randn(3, 4), (slice(0, 2), slice(1, 3))),
        (np.random.randn(8, 1), (np.array([0, 2, 4]),)),
        (np.random.randn(2, 3), (np.array([False, True]), slice(0, 2))),
        (np.random.randn(3), (1,)),
    ],
)
def test_getitem_forward(x, index):
    t1 = Tensor(x, requires_grad=True)
    result = t1[index]

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch[index]
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, index",
    [
        (np.random.randn(2, 3), (slice(0, 1),)),
        (np.random.randn(3, 4), (slice(0, 2), slice(1, 3))),
        (np.random.randn(8, 1), (np.array([0, 2, 4]),)),
        (np.random.randn(2, 3), (np.array([False, True]), slice(0, 2))),
        (np.random.randn(3), (1,)),
    ],
)
def test_getitem_backward(x, index):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano[index]
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch[index]
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for SetItem
@pytest.mark.parametrize(
    "x, index, value",
    [
        (np.random.randn(2, 3), (slice(0, 1),), np.random.randn(1, 3)),
        (np.random.randn(3, 4), (slice(0, 2), slice(1, 3)), np.random.randn(2, 2)),
        (np.random.randn(8, 1), (np.array([0, 2, 4]),), np.random.randn(3, 1)),
        (np.random.randn(2, 3), (np.array([False, True]), slice(0, 2)), np.random.randn(1, 2)),
        (np.random.randn(3), (1,), np.random.randn(1)),
    ],
)
def test_setitem_forward(x, index, value):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(value, requires_grad=True)
    out_nano = t1.clone()
    out_nano[index] = t2
    print(t1)

    x_torch = torch.tensor(x, requires_grad=True).clone()
    value_torch = torch.tensor(value, requires_grad=True)
    print(value_torch)
    result_torch = x_torch.clone()
    result_torch[index] = value_torch
    print(result_torch)

    assert_tensors_equal(out_nano, result_torch)

@pytest.mark.parametrize(
    "x, index, value",
    [
        (np.random.randn(2, 3), (slice(0, 1),), np.random.randn(1, 3)),
        (np.random.randn(3, 4), (slice(0, 2), slice(1, 3)), np.random.randn(2, 2)),
        (np.random.randn(8, 1), (np.array([0, 2, 4]),), np.random.randn(3, 1)),
        (np.random.randn(2, 3), (np.array([False, True]), slice(0, 2)), np.random.randn(1, 2)),
        (np.random.randn(3), (1,), np.random.randn(1)),
    ],
)
def test_setitem_backward(x, index, value, capsys):
    x_nano = Tensor(x, requires_grad=True)
    value_nano = Tensor(value, requires_grad=True)

    out_nano = x_nano.clone()
    out_nano[index] = value_nano

    out_nano.sum().backward()
    captured = capsys.readouterr()

    x_torch = torch.tensor(x, requires_grad=True)
    value_torch = torch.tensor(value, requires_grad=True)


    result_torch = x_torch.clone()
    result_torch[index] = value_torch
    result_torch.sum().backward()

    print("Captured stdout:", captured.out)

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(value_nano, value_torch)

# Test for Where
@pytest.mark.parametrize(
    "condition, x, y",
    [
        (np.random.randn(2, 3) > 0, np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4) > 0, np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(8, 1) > 0, np.random.randn(8, 1), np.random.randn(8, 1)),
        (np.random.randn(2, 3) > 0, np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3) > 0, np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_where_forward(condition, x, y):
    cond_t = Tensor(condition, requires_grad=False)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1.where(cond_t, t2)

    cond_torch = torch.tensor(condition, requires_grad=False)
    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.where(cond_torch, x_torch, y_torch)

    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "condition, x, y",
    [
        (np.random.randn(2, 3) > 0, np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4) > 0, np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(8, 1) > 0, np.random.randn(8, 1), np.random.randn(8, 1)),
        (np.random.randn(2, 3) > 0, np.random.randn(2, 3), np.random.randn(3)),
        (np.random.randn(3) > 0, np.random.randn(3), np.random.randn(3, 3)),
    ],
)
def test_where_backward(condition, x, y):
    cond_t = Tensor(condition, requires_grad=False)
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano.where(cond_t, y_nano)
    result.sum().backward()

    cond_torch = torch.tensor(condition, requires_grad=False)
    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = torch.where(cond_torch, x_torch, y_torch)
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)