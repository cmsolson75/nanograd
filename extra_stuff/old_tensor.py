import numpy as np
import math


class Tensor:
    def __init__(self, data, requires_grad=False, _prev=(), dtype=np.float32):
        self.data = (data if isinstance(data, np.ndarray) else np.array(data)).astype(
            dtype=dtype
        )
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_prev)
        self.grad_fn = str()  # for debug
        self.requires_grad = requires_grad

    # --------------
    # Creational ops
    # --------------
    @staticmethod
    def calculate_fan_in(weight):
        """
        Calculates the fan-in (number of incoming connections) for a weight tensor.

        Args:
            weight (np.ndarray or torch.Tensor): The weight tensor of a linear layer.

        Returns:
            int: The fan-in (number of incoming connections).
        """
        if len(weight.shape) == 1:
            return weight.shape[0]
        else:
            return weight.shape[1]

    @classmethod
    def kaiming_uniform(cls, in_dims, out_dims, gain: float = math.sqrt(5), **kwargs):
        """
        Kaiming Uniform initialization (He initialization) for weight matrices.

        Args:
            in_dims (int): Number of input dimensions (number of input units).
            out_dims (int): Number of output dimensions (number of neurons in the layer).
            gain (float): The scaling factor (set to sqrt(5) for a=sqrt(5) behavior).

        Returns:
            numpy.ndarray: The initialized weights.
        """

        fan_in = cls.calculate_fan_in(
            np.zeros((out_dims, in_dims))
        )  # Create a dummy weight for fan calculation

        bound = (
            np.sqrt(3.0) * np.sqrt(2.0 / (1 + gain**2)) / np.sqrt(fan_in)
        )  # Compute the correct boundary for uniform distribution
        data = np.random.uniform(
            -bound, bound, size=(in_dims, out_dims)
        )  # Notice the size parameter
        return cls(data, **kwargs)

    def kaiming_normal(self):
        pass

    def xavier_uniform(self):
        pass

    def xavier_normal(self):
        pass

    @classmethod
    def zeros(cls, shape: tuple, **kwargs):
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def ones(cls, shape: tuple, **kwargs):
        return cls(np.ones(shape), **kwargs)

    @classmethod
    def eye(cls, N, M=None, k=0, **kwargs):
        M = N if M is None else M

        return cls(np.eye(N, M, k), **kwargs)

    @classmethod
    def randn(cls, shape: tuple, **kwargs):
        return cls(np.random.randn(*shape), **kwargs)

    @classmethod
    def randint(cls, low, high=None, size=None, **kwargs):
        return cls(np.random.randint(low=low, high=high, size=size), **kwargs)

    @classmethod
    def uniform(cls, low, high=None, size=None, **kwargs):
        return cls(np.random.uniform(low=low, high=high, size=size), **kwargs)

    @classmethod
    def normal(cls, loc, scale=1.0, size=None, **kwargs):
        return cls(np.random.normal(loc=loc, scale=scale, size=size), **kwargs)

    @classmethod
    def arange(cls, start, stop, step, **kwargs):
        return cls(np.arange(start, stop, step), **kwargs)

    # ---------

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndims(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def numpy(self):
        return self.data

    @property
    def size(self):
        return self.data.size

    def item(self):
        return self.data.item()

    # ----------
    # Binary Ops
    # ----------

    def __matmul__(self, other):
        assert (
            self.shape[1] == other.shape[0]
        ), f"{self.shape}, {other.shape}: Incompatible shapes for matrix multiplication."
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), _prev=(self, other))

        def _backward():
            # lazy grad init for efficiancy
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.dot(out.grad, other.data.T)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += np.dot(self.data.T, out.grad)

        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "DotBackward"

        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = out.grad
                if grad_self.shape != self.shape:
                    axis = tuple(range(len(grad_self.shape) - len(self.shape)))
                    grad_self = grad_self.sum(axis=axis)
                    for i, dim in enumerate(self.shape):
                        if dim == 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad += grad_self

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad
                if grad_other.shape != other.shape:
                    axis = tuple(range(len(grad_other.shape) - len(other.shape)))
                    grad_other = grad_other.sum(axis=axis)
                    for i, dim in enumerate(other.shape):
                        if dim == 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad += grad_other

        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "AddBackward"

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other.data * out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(self.data)
                other.grad += self.data * out.grad

        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MulBackward"

        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Tensor(np.power(self.data, other), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (other * np.power(self.data, (other - 1))) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "PowBackward"
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return self / other

    # ---------
    # Unary ops
    # ---------

    # ********activations********

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (out.data > 0) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "ReluBackward"
        return out

    def tanh(self):
        # epsilon = 1e-8 # Stablizer
        n = self.data
        t = (np.exp(2 * n) - 1) / (np.exp(2 * n) + 1)  # Calc tanh
        out = Tensor(t, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 - t**2) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "TanhBackward"
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + np.exp(-x))
        out = Tensor(s, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.data * (1 - out.data) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "SigmoidBackward"
        return out

    def softmax(self, axis=-1):
        # for stability
        m = self - self.max(axis=axis, keepdims=True)
        e = m.exp()
        return e / e.sum()

    def log_softmax(self, axis=-1):
        max_tensor = self.max(axis=axis, keepdims=True)
        m = self - max_tensor
        e = m.exp()
        s = e.sum(axis=axis, keepdims=True)
        return m - s.log()

    def swish(self):
        # I think this will work
        return self * self.sigmoid()

    def gelu(self):
        return 0.5 * self * (1 + 0.797884560803 * (self + 0.044715 * self**2)).tanh()

    # ******standard*******
    def transpose(self):
        out = Tensor(np.transpose(self.data), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.T

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "TransposeBackward"

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.data * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "ExpBackward"
        return out

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("Logarithm undefined for non-positive values.")
        out = Tensor(np.log(self.data), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 / self.data) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "LogBackward"
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.cos(self.data) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "SinBackward"
        return out

    def square(self):
        return self**2

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.sign(self.data) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "AbsBackward"

        return out

    def round(self):
        rounded_data = np.round(self.data)
        out = Tensor(rounded_data, _prev=[self])

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # For the gradient of the rounding function, we use a sub-gradient approach
                self.grad += out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "RoundBackward"

        return out

    # ---------
    # Ternary ops
    # ---------

    def where(self, condition, y):
        """
        Element-wise selection from self or y depending on the condition.

        Args:
            condition (Tensor): Condition tensor, same shape as self.
            y (Tensor): Alternative values, same shape as self.

        Returns:
            Tensor: Resulting tensor with elements from self where condition is True, else from y.
        """
        if not isinstance(condition, Tensor):
            condition = Tensor(condition)
        if not isinstance(y, Tensor):
            y = Tensor(y)

        out = Tensor(
            np.where(condition.data, self.data, y.data),
            requires_grad=self.requires_grad or y.requires_grad,
            _prev=(self, condition, y),
        )

        def _backward():
            if condition.requires_grad or self.requires_grad or y.requires_grad:
                if out.grad is None:
                    return
                if condition.requires_grad:
                    if condition.grad is None:
                        condition.grad = np.zeros_like(condition.data)
                    # Gradient of where w.r.t. condition is zero (discontinuous function)
                    condition.grad += np.zeros_like(condition.data)
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += out.grad * condition.data
                if y.requires_grad:
                    if y.grad is None:
                        y.grad = np.zeros_like(y.data)
                    y.grad += out.grad * (1 - condition.data)

        if out.requires_grad:
            out._backward = _backward
            out.grad_fn = "WhereBackward"

        return out

    # ----------
    # Reduce ops
    # ----------

    def sum(self, axis=None, keepdims=False):
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = np.ones_like(self.data)
                self.grad += grad * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "SumBackward"

        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # distribute grad with scale 1/n
                grad = np.full(self.data.shape, 1 / self.data.size)
                self.grad += grad * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MeanBackward"

        return out

    def std(self):
        # Make this back propable
        return self.data.std()

    def max(self, axis=None, keepdims=False):
        max_value = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_value, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                max_value = np.max(self.data, axis=axis, keepdims=keepdims)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axis is None:
                    grad_mask = self.data == max_value
                else:
                    if max_value.ndim == 1:
                        max_value = np.expand_dims(max_value, axis=axis)
                        out.grad = np.expand_dims(out.grad, axis=axis)
                    grad_mask = (
                        self.data == max_value
                    )  # np.expand_dims(max_value, axis=axis)
                self.grad += grad_mask * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MaxBackward"

        return out

    def min(self, axis=None, keepdims=False):
        min_value = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(min_value, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            min_value = np.min(self.data, axis=axis, keepdims=keepdims)
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if axis is None:
                    grad_mask = self.data == min_value
                else:
                    if min_value.ndim == 1:
                        min_value = np.expand_dims(min_value, axis=axis)
                        out.grad = np.expand_dims(out.grad, axis=axis)
                    grad_mask = self.data == min_value
                self.grad += grad_mask * out.grad

        if self.requires_grad:
            out._backward = _backward
            out.grad_fn = "MinBackward"

        return out

    def maximum(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full_like(self.data, other))
        out = Tensor(
            np.maximum(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = out.grad * (self.data >= other.data)
                self.grad += grad_self
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad * (self.data < other.data)
                other.grad += grad_other

        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MaximumBackward"

        return out

    def minimum(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.full_like(self.data, other))
        out = Tensor(
            np.minimum(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _prev=(self, other),
        )

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self = out.grad * (self.data <= other.data)
                self.grad += grad_self
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad_other = out.grad * (self.data > other.data)
                other.grad += grad_other

        if self.requires_grad or other.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MinimumBackward"

        return out

    def clip(self, min_, max_):
        return self.maximum(min_).minimum(max_)

    def flatten(self, start_dim, end_dim=-1):
        x = self.data
        if end_dim == -1:
            end_dim = len(x.shape) - 1
        # new shape. This is hard to read but trust in the process it works
        shape = (
            x.shape[:start_dim]
            + (np.prod(x.shape[start_dim : end_dim + 1]),)
            + x.shape[end_dim + 1 :]
        )
        return self.reshape(shape)

    # ------------
    # Movement Ops
    # ------------

    def reshape(self, *shape):
        reshaped_data = self.data.reshape(*shape)
        out = Tensor(reshaped_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # Insure grad is reshaped to pass through this op
                self.grad += out.grad.reshape(self.data.shape)

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "ReshapeBackward"
        return out

    def pad(self, pad_width, constant_values=0):
        padded_output = np.pad(
            self.data,
            pad_width=pad_width,
            mode="constant",
            constant_values=constant_values,
        )

        out = Tensor(padded_output, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                slices = tuple(
                    slice(pad[0], -pad[1] if pad[1] > 0 else None) for pad in pad_width
                )
                self.grad += out.grad[slices]

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "PadBackward"

        return out

    def shrink(self, shrink_dims):
        slices = tuple(
            slice(start, -end if end > 0 else None) for start, end in shrink_dims
        )
        shrunk_data = self.data[slices]
        out = Tensor(shrunk_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                pad_width = [(start, end) for start, end in shrink_dims]
                self.grad += np.pad(
                    out.grad, pad_width, mode="constant", constant_values=0
                )

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "ShrinkBackward"

        return out

    def squeeze(self, axis=None):
        squeezed_data = np.squeeze(self.data, axis=axis)
        out = Tensor(squeezed_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                expanded_grad = np.reshape(out.grad, self.data.shape)
                self.grad += expanded_grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "SqueezeBackward"

        return out

    def unsqueeze(self, axis):
        unsqueezed_data = np.expand_dims(self.data, axis)
        out = Tensor(unsqueezed_data, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                reduced_grad = np.squeeze(out.grad, axis=axis)
                self.grad += reduced_grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "UnsqueezeBackward"

        return out

    # ----------
    # Functional nn ops
    # ----------
    # conv2d etc

    # ----------
    # Utility Ops
    # ----------

    def argmax(self, axis=None):
        """Returns the indices of the maximum values along an axis."""
        indices = np.argmax(self.data, axis=axis)
        return Tensor(indices, _prev=(self,), dtype=np.int64)

    def argmin(self, axis=None):
        """Returns the indices of the minimum values along an axis."""
        indices = np.argmin(self.data, axis=axis)
        return Tensor(indices, _prev=(self,), dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data)
        return Tensor(self.data == other)

    def __hash__(self):
        return id(self)

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = index.data

        result = self.data[index]
        out = Tensor(result, _prev=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_output = np.zeros_like(self.data)

                if isinstance(index, slice):
                    grad_output[index] = out.grad
                elif isinstance(index, np.ndarray):
                    if index.dtype == np.bool or np.issubdtype(index.dtype, np.integer):
                        np.add.at(grad_output, index, out.grad)
                    else:
                        raise NotImplementedError(
                            f"Unsupported indexing type for gradients: {index.dtype}"
                        )
                elif isinstance(index, tuple):
                    grad_output[index] = out.grad
                else:
                    raise NotImplementedError(
                        f"Unsupported indexing type for gradients: {type(index)}"
                    )

                # Ensure grad_output is properly broadcasted to match the shape of self.data
                if grad_output.shape != self.data.shape:
                    for i in range(len(self.data.shape)):
                        if self.data.shape[i] != grad_output.shape[i]:
                            grad_output = np.sum(grad_output, axis=i, keepdims=True)

                self.grad += grad_output

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "IndexBackward"

        return out

    def __format__(self, format_spec):
        """Format the Tensor's data according to the format_spec."""
        if format_spec == "":
            return str(self.data)
        else:
            formatted_data = np.array2string(
                self.data,
                formatter={
                    "float_kind": lambda x: format(x, format_spec),
                    "int_kind": lambda x: format(x, format_spec),
                },
            )
            return formatted_data

    def __repr__(self):
        return f"tensor({self.data})"

    # ---------
    # Backwards
    # ---------

    def _topological_sort(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        return topo

    def _compute_gradients(self, topo_sorted):
        self.grad = np.ones_like(self.data)
        for node in reversed(topo_sorted):
            node._backward()

    def backward(self):
        topo_sorted = self._topological_sort()
        self._compute_gradients(topo_sorted)

# Dont know if you need this: Maybe refactor in later to the NN for implicet grad init.
class Parameter(Tensor):
    """
    A special kind of tensor that is to be considered a module parameter.
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        """
        Initializes a parameter.

        Args:
            data (np.ndarray): The data for the parameter.
            requires_grad (bool): If True, gradients will be calculated for this parameter. Default is True.
        """
        super().__init__(data, requires_grad=requires_grad)