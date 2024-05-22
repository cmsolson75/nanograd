import pytest
import torch
import numpy as np
from tensor import Tensor  # Replace with the actual import
from nn import NLLLoss  # Replace with the actual import

EPSILON = 1e-6  # Tolerance for floating point comparison

def assert_tensors_equal(tensor1, tensor2):
    np.testing.assert_allclose(tensor1.data, tensor2.detach().numpy(), rtol=EPSILON, atol=EPSILON)

def assert_gradients_equal(tensor1, tensor2):
    if tensor1.grad is None or tensor2.grad is None:
        raise ValueError("One of the tensors does not have a gradient. Ensure retain_grad() is called on non-leaf tensors.")
    np.testing.assert_allclose(tensor1.grad, tensor2.grad.detach().numpy(), rtol=EPSILON, atol=EPSILON)

@pytest.mark.parametrize("logits, targets", [
    (np.random.randn(5, 10), np.random.randint(0, 10, size=(5,))),
    (np.random.randn(8, 4), np.random.randint(0, 4, size=(8,))),
])
def test_nll_loss(logits, targets):
    print(logits.shape, targets.shape)
    logits_np = np.array(logits, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.int64)
    
    # Log-softmax using PyTorch to ensure consistency
    logits_torch = torch.tensor(logits_np, requires_grad=True)
    log_probs_torch = torch.nn.functional.log_softmax(logits_torch, dim=1)
    targets_torch = torch.tensor(targets_np, dtype=torch.int64)
    
    # Retain gradients for non-leaf tensors
    log_probs_torch.retain_grad()
    
    # PyTorch NLLLoss
    nll_loss_torch = torch.nn.NLLLoss()
    loss_torch = nll_loss_torch(log_probs_torch, targets_torch)
    loss_torch.backward()
    print(loss_torch)
    print(log_probs_torch.grad)
    
    # Your custom NLLLoss
    log_probs_nano = Tensor(logits_np, requires_grad=True).log_softmax()
    targets = Tensor(targets_np, dtype=np.int64)  # Ensure targets are integers
    nll_loss = NLLLoss()
    loss = nll_loss(log_probs_nano, targets)
    print(loss)
    loss.backward()
    print(log_probs_nano.grad)
    
    assert_tensors_equal(loss, loss_torch)
    assert_gradients_equal(log_probs_nano, log_probs_torch)

if __name__ == "__main__":
    pytest.main()
