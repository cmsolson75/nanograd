from tensor_refactor import Function
import numpy as np



def broadcast_gradient(grad, shape):
    if grad.shape != shape:
        axis = tuple(range(len(grad.shape) - len(shape)))
        grad = grad.sum(axis=axis)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
    return grad

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        a_data = a.data
        b_data = b.data
        return a_data + b_data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = broadcast_gradient(grad_output, a.data.shape) if a.requires_grad else None
        grad_b = broadcast_gradient(grad_output, b.data.shape) if b.requires_grad else None
        return grad_a, grad_b
    

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        a_data = a.data
        b_data = b.data
        return a_data * b_data
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b.data if a.requires_grad else None
        grad_b = grad_output * a.data if b.requires_grad else None

        if grad_a is not None:
            grad_a = broadcast_gradient(grad_a, a.data.shape)
        if grad_b is not None:
            grad_b = broadcast_gradient(grad_b, b.data.shape)

        return grad_a, grad_b