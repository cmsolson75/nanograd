import unittest
import torch
import numpy as np
from engine import Tensor

class TestTensorOperations(unittest.TestCase):
    def test_addition(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([4.0], requires_grad=True)
        c = a + b
        
        a_torch = torch.tensor([1.0], requires_grad=True)
        b_torch = torch.tensor([4.0], requires_grad=True)
        c_torch = a_torch + b_torch
        
        np.testing.assert_array_almost_equal(c.data, c_torch.data.numpy())
        c.backward()
        c_torch.backward()
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    def test_multiplication(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([4.0], requires_grad=True)
        c = a * b

        a_torch = torch.tensor([1.0], requires_grad=True)
        b_torch = torch.tensor([4.0], requires_grad=True)
        c_torch = a_torch * b_torch

        np.testing.assert_array_almost_equal(c.data, c_torch.data.numpy())
        c.backward()
        c_torch.backward()
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    def test_matrix_multiplication(self):
        a = Tensor([[1.0]], requires_grad=True)
        b = Tensor([[2.0]], requires_grad=True)
        c = a @ b

        a_torch = torch.tensor([[1.0]], requires_grad=True)
        b_torch = torch.tensor([[2.0]], requires_grad=True)
        c_torch = a_torch @ b_torch

        np.testing.assert_array_almost_equal(c.data, c_torch.data.numpy())
        c.backward()
        c_torch.backward()
        np.testing.assert_array_almost_equal(a.grad, a_torch.grad.numpy())
        np.testing.assert_array_almost_equal(b.grad, b_torch.grad.numpy())

    # Additional tests for functions like relu, sigmoid, etc.

if __name__ == '__main__':
    unittest.main()
