from tensor import Function
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


def create_grad_mask(
    a_data, extrema_value, grad_output, axis=None, keepdims=False, compare_op=np.equal
):
    if axis is None:
        grad_mask = compare_op(a_data, extrema_value)
    else:
        if not keepdims:
            extrema_value = np.expand_dims(extrema_value, axis=axis)
            grad_output = np.expand_dims(grad_output, axis=axis)
        grad_mask = compare_op(a_data, extrema_value)
    return grad_mask, grad_output


class Slice(Function):
    @staticmethod
    def forward(ctx, a, index):
        ctx.save_for_backward(a)
        ctx.save_attribute("index", index)
        return a.data[index]

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        index = ctx.saved_attributes["index"]
        grad_a = None
        if a.requires_grad:
            grad_a = np.zeros_like(a.data)
            if isinstance(index, slice):
                grad_a[index] = grad_output
            elif isinstance(index, np.ndarray):
                if index.dtype == np.bool or np.issubdtype(index.dtype, np.integer):
                    np.add.at(grad_a, index, grad_output)
                else:
                    raise NotImplementedError(
                        f"Unsupported indexing type for gradients: {index.dtype}"
                    )
            elif isinstance(index, tuple):
                grad_a[index] = grad_output
            else:
                raise NotImplementedError(
                    f"Unsupported indexing type for gradients: {type(index)}"
                )

        return grad_a


class Copy(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a.data.copy()

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        (a,) = ctx.saved_tensors
        grad_a = grad_output if a.requires_grad else None
        return grad_a


class SetItem(Function):
    @staticmethod
    def forward(ctx, a, index, value):
        ctx.save_for_backward(a, value)
        ctx.save_attribute("index", index)
        if np.isscalar(value.data) or value.data.size == 1:
            a.data[index] = value.data.item()
        else:
            a.data[index] = value.data

        return a.data

    @staticmethod
    def backward(ctx, grad_output):
        a, value = ctx.saved_tensors
        index = ctx.saved_attributes["index"]
        grad_a = None
        if a.requires_grad:
            grad_a = np.array(grad_output, copy=True)
            # Zero out the gradient in the modified regions
            grad_a[index] = 0

        # Initialize the gradient of the value with the gradient output at the specified index
        grad_value = (
            (
                grad_output[index]
                if isinstance(index, (slice, np.ndarray, tuple))
                else np.array([grad_output[index]])
            )
            if value.requires_grad
            else None
        )
        return grad_a, grad_value


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
        a_grad = np.exp(a.data) * grad_output if a.requires_grad else None
        return a_grad


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(a.data)
            result = np.where(
                a.data > 0, result, np.nan
            )  # Setting invalid results to NaN
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        with np.errstate(divide="ignore", invalid="ignore"):
            grad_input = np.zeros_like(a.data)
            positive_mask = a.data > 0
            zero_or_negative_mask = a.data <= 0

            # For positive values, the gradient is 1/x
            grad_input[positive_mask] = (
                grad_output[positive_mask] / a.data[positive_mask]
            )
            # Handle zero or negative values:
            # Instead of setting gradient to `inf` or `nan`, propagate the gradient output scaled by a large negative value.
            # This mimics the behavior observed in PyTorch, ensuring proper gradient propagation.
            epsilon = 1e-15  # for avoiding -1/0
            grad_input[zero_or_negative_mask] = grad_output[zero_or_negative_mask] * (
                -1.0 / np.abs(a.data[zero_or_negative_mask] + epsilon)
            )

        return grad_input


class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        with np.errstate(invalid="ignore"):
            result = np.sqrt(a.data)
            result = np.where(np.isnan(result), np.nan, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = None
        if a.requires_grad:
            with np.errstate(invalid="ignore"):
                grad_a = 0.5 / np.sqrt(a.data)
                grad_a = np.where(np.isnan(grad_a), np.nan, grad_a)
                grad_a = grad_a * grad_output  # chain rule
            grad_a = np.broadcast_to(grad_a, a.shape)
        return grad_a


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
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.abs(a.data)

    @staticmethod
    def backward(ctx, grad_outputs):
        (a,) = ctx.saved_tensors
        grad_a = np.sign(a.data) * grad_outputs if a.requires_grad else None
        return grad_a


class Round(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.round(a.data)

    @staticmethod
    def backward(ctx, grad_outputs):
        (a,) = ctx.saved_tensors
        grad_a = np.zeros_like(grad_outputs) if a.requires_grad else None
        return grad_a


class Ceil(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.ceil(a.data)

    @staticmethod
    def backward(ctx, grad_outputs):
        (a,) = ctx.saved_tensors
        grad_a = np.zeros_like(grad_outputs) if a.requires_grad else None
        return grad_a


class Floor(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.floor(a.data)

    @staticmethod
    def backward(ctx, grad_outputs):
        (a,) = ctx.saved_tensors
        grad_a = np.zeros_like(grad_outputs) if a.requires_grad else None
        return grad_a


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
        ctx.save_attribute("exponent", exponent)
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
        exponent = ctx.saved_attributes["exponent"]
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
        ctx.save_attribute("axis", axis)
        ctx.save_attribute("keepdims", keepdims)
        return np.sum(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.saved_attributes["axis"]
        keepdims = ctx.saved_attributes["keepdims"]
        a_grad = None
        if a.requires_grad:
            a_grad = np.ones_like(a.data)
            if axis is not None and not keepdims:
                grad_output = np.expand_dims(grad_output, axis=axis)
            a_grad *= grad_output
        return a_grad


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.save_attribute("axis", axis)
        ctx.save_attribute("keepdims", keepdims)
        return np.mean(a.data, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.saved_attributes["axis"]
        keepdims = ctx.saved_attributes["keepdims"]

        a_grad = None
        if a.requires_grad:
            a_grad = np.ones_like(a.data) / a.data.size

            if axis is not None and not keepdims:
                grad_output = np.expand_dims(grad_output, axis=axis)

            a_grad *= grad_output
        return a_grad


class Max(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        max_value = np.max(a.data, axis=axis, keepdims=keepdims)
        ctx.save_attribute("axis", axis)
        ctx.save_attribute("keepdims", keepdims)
        ctx.save_attribute("max_value", max_value)
        return max_value

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.saved_attributes["axis"]
        keepdims = ctx.saved_attributes["keepdims"]
        max_value = ctx.saved_attributes["max_value"]

        grad_mask, grad_output = create_grad_mask(
            a.data, max_value, grad_output, axis, keepdims, np.equal
        )

        grad_a = grad_mask * grad_output if a.requires_grad else None
        return grad_a


class Min(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        min_value = np.min(a.data, axis=axis, keepdims=keepdims)

        ctx.save_attribute("axis", axis)
        ctx.save_attribute("keepdims", keepdims)
        ctx.save_attribute("min_value", min_value)

        return min_value

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors

        axis = ctx.saved_attributes["axis"]
        keepdims = ctx.saved_attributes["keepdims"]
        min_value = ctx.saved_attributes["min_value"]

        grad_mask, grad_output = create_grad_mask(
            a.data, min_value, grad_output, axis, keepdims, np.equal
        )

        grad_a = grad_mask * grad_output if a.requires_grad else None
        return grad_a


class Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.maximum(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # Create masks for the gradients
        grad_a = (a.data >= b.data) * grad_output
        grad_b = (b.data >= a.data) * grad_output

        # Handle broadcasting by summing gradients where needed
        grad_a = broadcast_gradient(grad_a, a.data.shape) if a.requires_grad else None
        grad_b = broadcast_gradient(grad_b, b.data.shape) if b.requires_grad else None

        return grad_a, grad_b


class Minimum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.minimum(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # Create masks for the gradients
        grad_a = (a.data <= b.data) * grad_output
        grad_b = (b.data <= a.data) * grad_output

        # Handle broadcasting by summing gradients where needed
        grad_a = broadcast_gradient(grad_a, a.data.shape) if a.requires_grad else None
        grad_b = broadcast_gradient(grad_b, b.data.shape) if b.requires_grad else None

        return grad_a, grad_b


# ----------
# Movement Ops
# ----------


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        ctx.save_attribute("original_shape", a.shape)
        return a.data.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        original_shape = ctx.saved_attributes["original_shape"]
        grad_a = grad_output.reshape(original_shape) if a.requires_grad else None
        return grad_a


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axis=None):
        ctx.save_for_backward(a)
        ctx.save_attribute("axis", axis)
        return np.transpose(a.data, axis)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.saved_attributes["axis"]
        if axis is None:
            axis = tuple(range(a.data.ndim))[::-1]
        else:
            axis = np.argsort(axis)
        grad_a = np.transpose(grad_output, axis) if a.requires_grad else None
        return grad_a


# implement when you need Convs
class Pad(Function):
    pass


class Shrink(Function):
    pass


# ----------
# Ternary Ops
# ----------


class Where(Function):
    @staticmethod
    def forward(ctx, condition, x, y):
        ctx.save_attribute("condition", condition)
        ctx.save_for_backward(x, y)
        return np.where(condition.data, x.data, y.data)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        condition = ctx.saved_attributes["condition"]
        # init grads
        grad_x, grad_y = None, None

        if x.requires_grad:
            grad_x = np.where(condition.data, grad_output, 0)
            if grad_x.shape != x.data.shape:
                grad_x = broadcast_gradient(grad_x, x.data.shape)

        if y.requires_grad:
            grad_y = np.where(condition.data, 0, grad_output)
            if grad_y.shape != y.data.shape:
                grad_y = broadcast_gradient(grad_y, y.data.shape)

        return grad_x, grad_y


# ----------
# Functional NN Ops
# ----------


class Linear(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        return x.data @ weight.data + bias.data

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_output @ weight.data.T if x.requires_grad else None
        grad_weight = x.data.T @ grad_output if weight.requires_grad else None
        grad_bias = (
            grad_output.sum(axis=0, keepdims=True) if bias.requires_grad else None
        )
        return grad_x, grad_weight, grad_bias
