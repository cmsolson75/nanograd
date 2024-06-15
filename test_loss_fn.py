import pytest
import torch
import numpy as np
from tensor import Tensor
from nn import (
    NLLLoss,
    MSELoss,
    BCELoss,
    CrossEntropyLoss,
)

EPSILON = 1e-6  # Tolerance for floating point comparison


def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(
        tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


def assert_gradients_equal(tensor1, tensor2):
    if tensor1.grad is None or tensor2.grad is None:
        raise ValueError(
            "One of the tensors does not have a gradient. Ensure retain_grad() is called on non-leaf tensors."
        )
    np.testing.assert_allclose(
        tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON
    )


@pytest.mark.parametrize(
    "logits, targets",
    [
        (np.random.randn(5, 10), np.random.randint(0, 10, size=(5,))),
        (np.random.randn(8, 4), np.random.randint(0, 4, size=(8,))),
    ],
)
def test_nll_loss(logits, targets):
    logits_np = np.array(logits, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.int64)

    logits_torch = torch.tensor(logits_np, requires_grad=True)
    log_probs_torch = torch.nn.functional.log_softmax(logits_torch, dim=1)
    targets_torch = torch.tensor(targets_np, dtype=torch.int64)

    log_probs_torch.retain_grad()

    nll_loss_torch = torch.nn.NLLLoss()
    loss_torch = nll_loss_torch(log_probs_torch, targets_torch)
    loss_torch.backward()

    log_probs_nano = Tensor(logits_np, requires_grad=True).log_softmax()
    targets_nano = Tensor(targets_np, dtype=np.int64)
    nll_loss_nano = NLLLoss()
    loss_nano = nll_loss_nano(log_probs_nano, targets_nano)
    loss_nano.backward()

    assert_tensors_equal(loss_nano, loss_torch)
    assert_gradients_equal(log_probs_nano, log_probs_torch)


@pytest.mark.parametrize(
    "preds, targets",
    [
        (np.random.randn(5, 10), np.random.randn(5, 10)),
        (np.random.randn(8, 4), np.random.randn(8, 4)),
    ],
)
def test_mse_loss(preds, targets):
    preds_np = np.array(preds, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.float32)

    preds_torch = torch.tensor(preds_np, requires_grad=True)
    targets_torch = torch.tensor(targets_np)

    mse_loss_torch = torch.nn.MSELoss()
    loss_torch = mse_loss_torch(preds_torch, targets_torch)
    loss_torch.backward()

    preds_nano = Tensor(preds_np, requires_grad=True)
    targets_nano = Tensor(targets_np)
    mse_loss_nano = MSELoss()
    loss_nano = mse_loss_nano(preds_nano, targets_nano)
    loss_nano.backward()

    assert_tensors_equal(loss_nano, loss_torch)
    assert_gradients_equal(preds_nano, preds_torch)


@pytest.mark.parametrize(
    "preds, targets",
    [
        (np.random.rand(5, 1), np.random.randint(0, 2, size=(5, 1))),
        (np.random.rand(8, 1), np.random.randint(0, 2, size=(8, 1))),
    ],
)
def test_bce_loss(preds, targets):
    preds_np = np.array(preds, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.float32)

    preds_torch = torch.tensor(preds_np, requires_grad=True)
    targets_torch = torch.tensor(targets_np)

    bce_loss_torch = torch.nn.BCELoss()
    loss_torch = bce_loss_torch(preds_torch, targets_torch)
    loss_torch.backward()

    preds_nano = Tensor(preds_np, requires_grad=True)
    targets_nano = Tensor(targets_np)
    bce_loss_nano = BCELoss()
    loss_nano = bce_loss_nano(preds_nano, targets_nano)
    loss_nano.backward()

    assert_tensors_equal(loss_nano, loss_torch)
    assert_gradients_equal(preds_nano, preds_torch)


@pytest.mark.parametrize(
    "logits, targets",
    [
        (np.random.randn(5, 10), np.random.randint(0, 10, size=(5,))),
        (np.random.randn(8, 4), np.random.randint(0, 4, size=(8,))),
    ],
)
def test_cross_entropy_loss(logits, targets):
    logits_np = np.array(logits, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.int64)

    logits_torch = torch.tensor(logits_np, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.int64)

    ce_loss_torch = torch.nn.CrossEntropyLoss()
    loss_torch = ce_loss_torch(logits_torch, targets_torch)
    loss_torch.backward()

    logits_nano = Tensor(logits_np, requires_grad=True)
    targets_nano = Tensor(targets_np, dtype=np.int64)
    ce_loss_nano = CrossEntropyLoss()
    loss_nano = ce_loss_nano(logits_nano, targets_nano)
    loss_nano.backward()

    assert_tensors_equal(loss_nano, loss_torch)
    assert_gradients_equal(logits_nano, logits_torch)


if __name__ == "__main__":
    pytest.main()
