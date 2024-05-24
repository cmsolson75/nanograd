import pytest
import torch
import numpy as np
from tensor import Tensor  # Replace with the actual import

EPSILON = 1e-6  # Tolerance for floating point comparison


def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


def assert_gradients_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


@pytest.fixture
def setup_data():
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    return x, y


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(3)),  # Broadcasting case
        (np.random.randn(3), np.random.randn(3, 3)),  # Broadcasting case
        (np.random.randn(2, 3), np.random.randn(2, 3)),  # Regular case
        (np.random.randn(2, 3), np.random.randn(2, 3)),  # Regular case
    ],
)
def test_broadcast_addition(x, y):
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    y_torch = torch.tensor(y, requires_grad=True)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()  # Call sum() to get a scalar for backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),  # Ensure compatible shapes
        (np.random.randn(8, 1), np.random.randn(8, 4)),
    ],
)
def test_add(x, y):
    print(x.shape, y.shape)
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


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(3, 4)),
        (np.random.randn(5, 6), np.random.randn(6, 7)),
    ],
)
def test_matmul(x, y):
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


@pytest.mark.parametrize(
    "x, y",
    [
        (np.random.randn(2, 3), np.random.randn(2, 3)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
    ],
)
def test_mul(x, y):
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


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_relu(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.relu()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.relu(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_tanh(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.tanh()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.tanh(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_sigmoid(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.sigmoid()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.sigmoid(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_exp(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.exp()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.exp(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize(
    "x",
    [
        np.random.randn(2, 3) + 1.1,  # Avoid log(0)
        np.random.randn(4, 5) + 1.1,
        np.random.randn(2, 3) - 2,  # Test negative values
    ],
)
def test_log(x):
    t1 = Tensor(x, requires_grad=True)
    x_torch = torch.tensor(x, requires_grad=True)

    if np.any(x <= 0):
        with pytest.raises(ValueError):
            t1.log()
    else:
        result = t1.log()
        result.backward()

        result_torch = torch.log(x_torch)
        result_torch.sum().backward()

        assert_tensors_equal(result, result_torch)
        assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_sum(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.sum()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.sum()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_mean(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.mean()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.mean()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_max_og(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.max()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.max()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_min(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.min()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.min()
    result_torch.backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_reshape(x):
    t1 = Tensor(x, requires_grad=True)
    new_shape = (x.size // 2, 2)
    result = t1.reshape(new_shape)
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.reshape(new_shape)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize(
    "x, pad_width",
    [
        (np.random.randn(2, 3), ((1, 1), (2, 2))),
        (np.random.randn(4, 5), ((2, 2), (1, 1))),
    ],
)
def test_pad(x, pad_width):
    t1 = Tensor(x, requires_grad=True)
    result = t1.pad(pad_width)
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.nn.functional.pad(
        x_torch, [item for sublist in reversed(pad_width) for item in sublist]
    )
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


@pytest.mark.parametrize("x", [np.random.randn(2, 3), np.random.randn(4, 5)])
def test_flatten(x):
    t1 = Tensor(x, requires_grad=True)
    result = t1.flatten(0, -1)
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = x_torch.flatten(0, -1)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


def test_large_tensors():
    x = np.random.randn(1000, 1000)
    y = np.random.randn(1000, 1000)
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


def test_broadcasting_operations():
    x = np.random.randn(2, 3)
    y = np.random.randn(
        3,
    )  # This will be broadcast to (2, 3)
    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = t1 + t2
    result.backward()
    print(np.sum(t1.grad, (0)))

    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(y, requires_grad=True, dtype=torch.float32)
    result_torch = x_torch + y_torch
    result_torch.sum().backward()  # Call sum() to get a scalar for backward()

    print(f"t2: {t2.grad}, y_torch: {y_torch.grad}")
    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    assert_gradients_equal(t2, y_torch)


def test_chain_operations():
    x = np.random.randn(2, 3)
    t1 = Tensor(x, requires_grad=True)
    result = ((t1 + 2) * 3).exp().log().relu()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.relu(torch.log(torch.exp((x_torch + 2) * 3)))
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


def test_negative_values():
    x = np.random.randn(2, 3) - 5  # Shift mean to negative values
    t1 = Tensor(x, requires_grad=True)
    result = t1.relu()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.relu(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


def test_zeros():
    x = np.zeros((2, 3))
    t1 = Tensor(x, requires_grad=True)
    result = t1.exp()
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    result_torch = torch.exp(x_torch)
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


def test_mixed_operations():
    x = np.random.randn(2, 3)
    y = np.random.randn(3, 4)

    # convert to torch: for testing
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(y, requires_grad=True, dtype=torch.float32)
    result_torch = torch.relu(x_torch @ y_torch) + x_torch.sum()
    result_torch.sum().backward()

    t1 = Tensor(x, requires_grad=True)
    t2 = Tensor(y, requires_grad=True)
    result = (t1 @ t2).relu() + t1.sum()
    result.backward()
    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)
    print(f"t2: {t2.grad}, y_torch: {y_torch.grad}")
    assert_gradients_equal(t2, y_torch)


@pytest.mark.parametrize(
    "x, axis, keepdims",
    [
        (np.random.randn(2, 3), None, False),
        (np.random.randn(4, 5), None, False),
        (np.random.randn(2, 3), 0, False),
        (np.random.randn(2, 3), 1, False),
        (np.random.randn(4, 5), 0, True),
        (np.random.randn(4, 5), 1, True),
    ],
)
def test_max(x, axis, keepdims):
    t1 = Tensor(x, requires_grad=True)
    result = t1.max(axis=axis, keepdims=keepdims)
    result.backward()

    x_torch = torch.tensor(x, requires_grad=True)
    if axis is not None:
        result_torch = x_torch.max(dim=axis, keepdim=keepdims).values
    else:
        result_torch = x_torch.max()

    # For PyTorch, need to call .sum().backward() on the result to make it scalar
    result_torch.sum().backward()

    assert_tensors_equal(result, result_torch)
    assert_gradients_equal(t1, x_torch)


if __name__ == "__main__":
    pytest.main()
