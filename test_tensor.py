import pytest
import torch
import numpy as np
from tensor import Tensor

EPSILON = 1e-6  # Tolerance for floating point comparison
EPSILON_2 = 1e-3

def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


def assert_gradients_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )

def assert_tensors_equal_aprox_error(tensor1, tensor2):
    # for aprox functions(gelu) or unstable functions(tan)
    np.testing.assert_allclose(
        tensor1.data, tensor2.detach().numpy(), rtol=EPSILON_2, atol=EPSILON_2
    )

def assert_gradients_equal_aprox_error(tensor1, tensor2):
    # for aprox functions(gelu) or unstable functions(tan)
    np.testing.assert_allclose(
        tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON_2, atol=EPSILON_2
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
        np.random.randn(6, 6),
    ],
)
def test_log_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.log()
    result_torch = x_torch.log()

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.randn(6, 6),
    ],
)
def test_log_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.log()
    result_torch = x_torch.log()

    result_nano.sum().backward()
    result_torch.sum().backward()
    print("NANO GRAD: ", x_nano.grad)
    print("TORCH GRAD:", x_torch.grad.detach().numpy())

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

# Add the test for sqrt forward
@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,  # Adding 1 to avoid sqrt of zero
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.rand(6, 6) + 1,
        np.random.randn(8, 2) # test for neg
    ],
)
def test_sqrt_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.sqrt()
    result_torch = torch.sqrt(x_torch)

    assert_tensors_equal(result_nano, result_torch)


@pytest.mark.parametrize(
    "x",
    [
        np.random.rand(2, 3) + 1,
        np.random.rand(3, 4) + 1,
        np.random.rand(5, 1) + 1,
        np.random.rand(1, 5) + 1,
        np.random.rand(6, 6) + 1,
        np.random.randn(8, 2)
    ],
)
def test_sqrt_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.sqrt()
    result_torch = torch.sqrt(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

@pytest.mark.parametrize(
    "x, neg_slope",
    [
        (np.random.randn(2, 3), 0.01),
        (np.random.randn(3, 4), 0.02),
        (np.random.randn(5, 1), 0.03),
        (np.random.randn(1, 5), 0.04),
        (np.random.randn(6, 6), 0.05),
    ],
)
def test_leaky_relu_forward(x, neg_slope):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.leaky_relu(neg_slope=neg_slope)
    result_torch = torch.nn.functional.leaky_relu(x_torch, negative_slope=neg_slope)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, neg_slope",
    [
        (np.random.randn(2, 3), 0.01),
        (np.random.randn(3, 4), 0.02),
        (np.random.randn(5, 1), 0.03),
        (np.random.randn(1, 5), 0.04),
        (np.random.randn(6, 6), 0.05),
    ],
)
def test_leaky_relu_backward(x, neg_slope):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.leaky_relu(neg_slope=neg_slope)
    result_torch = torch.nn.functional.leaky_relu(x_torch, negative_slope=neg_slope)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(2, 3), -1),
        (np.random.randn(3, 4), 0),
        (np.random.randn(5, 1), 1),
        (np.random.randn(1, 5), -1),
        (np.random.randn(6, 6), 1),
    ],
)
def test_softmax_forward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.softmax(axis=axis)
    result_torch = torch.nn.functional.softmax(x_torch, dim=axis)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(2, 3), -1),
        (np.random.randn(3, 4), 0),
        (np.random.randn(5, 1), 1),
        (np.random.randn(1, 5), -1),
        (np.random.randn(6, 6), 1),
    ],
)
def test_softmax_backward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.softmax(axis=axis)
    result_torch = torch.nn.functional.softmax(x_torch, dim=axis)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for log_softmax
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(2, 3), -1),
        (np.random.randn(3, 4), 0),
        (np.random.randn(5, 1), 1),
        (np.random.randn(1, 5), -1),
        (np.random.randn(6, 6), 1),
    ],
)
def test_log_softmax_forward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.log_softmax(axis=axis)
    result_torch = torch.nn.functional.log_softmax(x_torch, dim=axis)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(2, 3), -1),
        (np.random.randn(3, 4), 0),
        (np.random.randn(5, 1), 1),
        (np.random.randn(1, 5), -1),
        (np.random.randn(6, 6), 1),
    ],
)
def test_log_softmax_backward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.log_softmax(axis=axis)
    result_torch = torch.nn.functional.log_softmax(x_torch, dim=axis)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for swish
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
def test_swish_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.swish()
    result_torch = x_torch * torch.sigmoid(x_torch)

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
def test_swish_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.swish()
    result_torch = x_torch * torch.sigmoid(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for exp2
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
def test_exp2_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.exp2()
    result_torch = torch.pow(2, x_torch)

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
def test_exp2_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.exp2()
    result_torch = torch.pow(2, x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for log2
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
def test_log2_forward(x):
    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.log2(x_torch)
    print(result_torch)

    x_nano = Tensor(x, requires_grad=True)
    result_nano = x_nano.log2()
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
def test_log2_backward(x):
    print("X:", x)
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    result_nano = x_nano.log2()
    print("Nano results", result_nano)
    result_torch = torch.log2(x_torch)
    print("Torch results", result_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    print("NANO GRAD: ", x_nano.grad)
    print("TORCH GRAD:", x_torch.grad.detach().numpy())

    assert_gradients_equal(x_nano, x_torch)


# Test for rsqrt
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
def test_rsqrt_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.rsqrt()
    result_torch = torch.rsqrt(x_torch)

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
def test_rsqrt_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.rsqrt()
    result_torch = torch.rsqrt(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for cos
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
def test_cos_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.cos()
    result_torch = torch.cos(x_torch)

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
def test_cos_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.cos()
    result_torch = torch.cos(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for tan
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
def test_tan_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    result_nano = x_nano.tan()
    result_torch = torch.tan(x_torch)

    assert_tensors_equal_aprox_error(result_nano, result_torch)


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
def test_tan_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.tan()
    result_torch = torch.tan(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal_aprox_error(x_nano, x_torch)


# Test for square
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
def test_square_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.square()
    result_torch = x_torch**2

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
def test_square_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.square()
    result_torch = x_torch**2

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for gelu
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
def test_gelu_forward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.gelu()
    result_torch = torch.nn.functional.gelu(x_torch)

    assert_tensors_equal_aprox_error(result_nano, result_torch)

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
def test_gelu_backward(x):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.gelu()
    result_torch = torch.nn.functional.gelu(x_torch)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal_aprox_error(x_nano, x_torch)

# Test for sub (as __sub__)
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
def test_sub_forward(x, y):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 - t2

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch - y_torch
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
def test_sub_backward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano - y_nano
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch - y_torch
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)

# Test for div (as __truediv__)
@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3) + 1),  # Adding 1 to avoid division by zero
        (np.random.randn(3, 4), np.random.randn(3, 4) + 1),
        (np.random.randn(8, 1), np.random.randn(8, 4) + 1),
        (np.random.randn(2, 3), np.random.randn(3) + 1),
        (np.random.randn(3), np.random.randn(3, 3) + 1),
    ],
)
def test_div_forward(x, y):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 / t2

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch / y_torch
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3) + 1),
        (np.random.randn(3, 4), np.random.randn(3, 4) + 1),
        (np.random.randn(8, 1), np.random.randn(8, 4) + 1),
        (np.random.randn(2, 3), np.random.randn(3) + 1),
        (np.random.randn(3), np.random.randn(3, 3) + 1),
    ],
)
def test_div_backward(x, y):
    x_nano = Tensor(x, requires_grad=True)
    y_nano = Tensor(y, requires_grad=True)
    result = x_nano / y_nano
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch / y_torch
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)
    assert_gradients_equal(y_nano, y_torch)

# Test for pow (as __pow__)
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
def test_pow_backward(x, exponent):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano ** exponent
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch ** exponent
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for clip
@pytest.mark.parametrize(
    "x, min_, max_",
    [
        (np.random.randn(2, 3), -0.5, 0.5),
        (np.random.randn(3, 4), -1.0, 1.0),
        (np.random.randn(5, 1), -0.1, 0.1),
        (np.random.randn(1, 5), 0, 1),
        (np.random.randn(6, 6), -2, 2),
    ],
)
def test_clip_forward(x, min_, max_):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.clip(min_, max_)
    result_torch = x_torch.clamp(min=min_, max=max_)

    assert_tensors_equal(result_nano, result_torch)

@pytest.mark.parametrize(
    "x, min_, max_",
    [
        (np.random.randn(2, 3), -0.5, 0.5),
        (np.random.randn(3, 4), -1.0, 1.0),
        (np.random.randn(5, 1), -0.1, 0.1),
        (np.random.randn(1, 5), 0, 1),
        (np.random.randn(6, 6), -2, 2),
    ],
)
def test_clip_backward(x, min_, max_):
    x_nano = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    result_nano = x_nano.clip(min_, max_)
    result_torch = x_torch.clamp(min=min_, max=max_)

    result_nano.sum().backward()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for view (alias for reshape)
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
def test_view_forward(x, shape):
    t1 = Tensor(x, requires_grad=True)
    result = t1.view(*shape)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.view(*shape)
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
def test_view_backward(x, shape):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.view(*shape)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.view(*shape)
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for squeeze
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(1, 3), None),
        (np.random.randn(3, 1), 1),
        (np.random.randn(1, 4, 1), None),
        (np.random.randn(2, 1, 3), 1),
    ],
)
def test_squeeze_forward(x, axis):
    t1 = Tensor(x, requires_grad=True)
    result = t1.squeeze(axis)

    x_torch = torch.tensor(x, requires_grad=True)
    if axis is not None:
        result_torch = x_torch.squeeze(dim=axis)
    else:
        result_torch = x_torch.squeeze()
    assert_tensors_equal(result, result_torch)

# Test for squeeze
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(1, 3), None),
        (np.random.randn(3, 1), 1),
        (np.random.randn(1, 4, 1), None),
        (np.random.randn(2, 1, 3), 1),
    ],
)
def test_squeeze_backward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.squeeze(axis)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    if axis is not None:
        result_torch = x_torch.squeeze(dim=axis)
    else:
        result_torch = x_torch.squeeze()
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for unsqueeze
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(3), 0),
        (np.random.randn(3, 4), 1),
        (np.random.randn(5, 1), 2),
        (np.random.randn(1, 5), 0),
        (np.random.randn(6, 6), 1),
    ],
)
def test_unsqueeze_forward(x, axis):
    t1 = Tensor(x, requires_grad=True)
    result = t1.unsqueeze(axis)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.unsqueeze(dim=axis)
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(3), 0),
        (np.random.randn(3, 4), 1),
        (np.random.randn(5, 1), 2),
        (np.random.randn(1, 5), 0),
        (np.random.randn(6, 6), 1),
    ],
)
def test_unsqueeze_backward(x, axis):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.unsqueeze(axis)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.unsqueeze(dim=axis)
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)

# Test for flatten
@pytest.mark.parametrize(
    "x, start_dim, end_dim",
    [
        (np.random.randn(2, 3, 4), 1, -1),
        (np.random.randn(3, 4, 5), 0, 1),
        (np.random.randn(5, 1), 0, -1),
        (np.random.randn(1, 5, 6), 1, 2),
        (np.random.randn(6, 6), 0, 1),
    ],
)
def test_flatten_forward(x, start_dim, end_dim):
    t1 = Tensor(x, requires_grad=True)
    result = t1.flatten(start_dim, end_dim)

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.flatten(x_torch, start_dim=start_dim, end_dim=end_dim)
    assert_tensors_equal(result, result_torch)

@pytest.mark.parametrize(
    "x, start_dim, end_dim",
    [
        (np.random.randn(2, 3, 4), 1, -1),
        (np.random.randn(3, 4, 5), 0, 1),
        (np.random.randn(5, 1), 0, -1),
        (np.random.randn(1, 5, 6), 1, 2),
        (np.random.randn(6, 6), 0, 1),
    ],
)
def test_flatten_backward(x, start_dim, end_dim):
    x_nano = Tensor(x, requires_grad=True)
    result = x_nano.flatten(start_dim, end_dim)
    result.sum().backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.flatten(x_torch, start_dim=start_dim, end_dim=end_dim)
    result_torch.sum().backward()

    assert_gradients_equal(x_nano, x_torch)


# Test for argmax
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(3, 4), None),
        (np.random.randn(3, 4), 0),
        (np.random.randn(3, 4), 1),
        (np.random.randn(2, 3, 4), 2),
    ],
)
def test_argmax(x, axis):
    t = Tensor(x)
    result = t.argmax(axis=axis)

    x_torch = torch.tensor(x)
    result_torch = x_torch.argmax(dim=axis)
    assert_tensors_equal(result, result_torch)

# Test for argmin
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.random.randn(3, 4), None),
        (np.random.randn(3, 4), 0),
        (np.random.randn(3, 4), 1),
        (np.random.randn(2, 3, 4), 2),
    ],
)
def test_argmin(x, axis):
    t = Tensor(x)
    result = t.argmin(axis=axis)

    x_torch = torch.tensor(x)
    result_torch = x_torch.argmin(dim=axis)
    assert_tensors_equal(result, result_torch)

# Test for var
@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(3, 4).astype(np.float32), None, False),
        (np.random.randn(3, 4).astype(np.float32), 0, False),
        (np.random.randn(3, 4).astype(np.float32), 1, True),
        (np.random.randn(2, 3, 4).astype(np.float32), 2, True),
    ],
)
def test_var(x, axis, keepdims):
    t = Tensor(x)
    result = t.var(axis=axis, keepdims=keepdims, unbiased=False)

    x_torch = torch.tensor(x, dtype=torch.float32)
    result_torch = x_torch.var(dim=axis, keepdim=keepdims, unbiased=False)
    assert_tensors_equal(result, result_torch)

# Test for std
@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(3, 4).astype(np.float32), None, False),
        (np.random.randn(3, 4).astype(np.float32), 0, False),
        (np.random.randn(3, 4).astype(np.float32), 1, True),
        (np.random.randn(2, 3, 4).astype(np.float32), 2, True),
    ],
)
def test_std(x, axis, keepdims):
    t = Tensor(x)
    result = t.std(axis=axis, keepdims=keepdims, unbiased=False)

    x_torch = torch.tensor(x, dtype=torch.float32)
    result_torch = x_torch.std(dim=axis, keepdim=keepdims, unbiased=False)
    assert_tensors_equal(result, result_torch)