from tensor_refactor import Function
import numpy as np

# Need to add typing

# ----------
# Util Ops
# ----------


def broadcast_gradient(grad, shape):
    if grad.shape != shape:
        axis = tuple(range(len(grad.shape) - len(shape)))
        grad = grad.sum(axis=axis)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
    return grad


class GetItem(Function):
    pass


# ----------
# Unary Ops
# ----------
class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return -a.data

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = -grad_output if a.requires_grad else None
        return grad_a


class Reciprocal(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return 1 / a.data

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = -grad_output / (a.data**2) if a.requires_grad else None
        return grad_a


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.exp(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a_grad = a.data * grad_output if a.requires_grad else None
        return a_grad


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if np.any(a.data <= 0):
            raise ValueError(f"Logarithm undefined for non-positive values {a.data}")
        ctx.save_for_backward(a)
        return np.log(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a_grad = (1 / a.data) * grad_output if a.requires_grad else None
        return a_grad


class Sin(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.sin(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = np.cos(a.data) * grad_output if a.requires_grad else None
        return grad_a


class Abs(Function):
    pass


class Round(Function):
    pass


class Ceil(Function):
    pass


class Floor(Function):
    pass


# Activations
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.tanh(a.data)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = grad_output * (1 - np.tanh(a.data) ** 2) if a.requires_grad else None
        return grad_a


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        sigmoid = 1 / (1 + np.exp(-a.data))
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        sigmoid = 1 / (1 + np.exp(-a.data))
        grad_a = grad_output * sigmoid * (1 - sigmoid) if a.requires_grad else None
        return grad_a


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.maximum(0, a.data)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = (a.data > 0) * grad_output if a.requires_grad else None
        return grad_a


# ----------
# Binary Ops
# ----------
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data + b.data

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = (
            broadcast_gradient(grad_output, a.data.shape) if a.requires_grad else None
        )
        grad_b = (
            broadcast_gradient(grad_output, b.data.shape) if b.requires_grad else None
        )
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.data * b.data

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


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        assert (
            a.shape[1] == b.shape[0]
        ), f"{a.shape}, {b.shape}: Incompatible shapes for matrix multiplication."
        ctx.save_for_backward(a, b)
        return np.dot(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.dot(grad_output, b.data.T) if a.requires_grad else None
        grad_b = np.dot(a.data.T, grad_output) if b.requires_grad else None

        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a)
        ctx.exponent = exponent
        # Check for invalid values and set to nan
        # issues with a**-exponent: this creates a sqrt
        # nan = complex number I dont wanna deal with
        # Also how pytorch handles it
        with np.errstate(invalid="ignore"):
            result = np.power(a.data, exponent)
            result = np.where(np.isnan(result), np.nan, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        exponent = ctx.exponent
        # Compute gradient: nan implementation like pytorch
        grad_a = None
        if a.requires_grad:
            with np.errstate(invalid="ignore"):
                grad_a = exponent * np.power(a.data, (exponent - 1))
                grad_a = np.where(np.isnan(grad_a), np.nan, grad_a)
                grad_a = grad_a * grad_output

            # Handle broadcasting correctly if needed
            grad_a = np.broadcast_to(grad_a, a.shape)

        return grad_a


# ----------
# Reduction Ops
# ----------


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        # special case for reduction methods
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a_grad = np.ones_like(a.data)

        if ctx.axis is not None and not ctx.keepdims:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)
        a_grad *= grad_output
        return a_grad


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.mean(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        a_grad = np.ones_like(a.data) / a.data.size

        if ctx.axis is not None and not ctx.keepdims:
            grad_output = np.expand_dims(grad_output, axis=ctx.axis)

        a_grad *= grad_output
        return a_grad


class Max(Function):
    pass


class Min(Function):
    pass


class Maximum(Function):
    pass


class Minimum(Function):
    pass


# ----------
# Movement Ops
# ----------


class Reshape(Function):
    pass


class Transpose(Function):
    pass


class Pad(Function):
    pass


class Shrink(Function):
    pass


# ----------
# Ternary Ops
# ----------


class Where(Function):
    pass
