import numpy as np
import math

import initializers as init
# Need to add typing


class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        output = cls.forward(ctx, *args, **kwargs)
        requires_grad = any(
            arg.requires_grad for arg in args if isinstance(arg, Tensor)
        )
        result = Tensor(output, requires_grad=requires_grad)
        result._ctx = ctx
        result._op = cls
        if result.requires_grad:
            # pytorch api & debug
            result.grad_fn = f"{cls.__name__}Backward"
        return result


# import at this point to stop circular imports
import functions as F


class Context:
    def __init__(self):
        self.saved_tensors = []
        self.saved_attributes = {}

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def save_attribute(self, name, value):
        self.saved_attributes[name] = value


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = (data if isinstance(data, np.ndarray) else np.array(data)).astype(
            dtype=dtype
        )
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None
        self._op = None
        self.grad_fn = None

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

    @property
    def T(self):
        return self.transpose()

    def item(self):
        return self.data.item()

    # Backward methods
    def backward(self, grad_output=None):
        if grad_output is None:
            if self.ndims > 0:
                raise ValueError("grad_output must be specified for non-scalar outputs")
            grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
        else:
            grad_output = Tensor._ensure_tensor(grad_output)

        self.grad = grad_output.data

        topo_sorted = self._topological_sort()
        self._compute_gradients(topo_sorted)

    def _topological_sort(self):
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor._ctx is not None:
                    for parent in tensor._ctx.saved_tensors:
                        if isinstance(parent, Tensor):
                            build_topo(parent)
                topo.append(tensor)

        build_topo(self)
        return topo

    def _compute_gradients(self, topo_sorted):
        for tensor in reversed(topo_sorted):
            if tensor._ctx is not None and tensor._op is not None:
                grads = tensor._op.backward(tensor._ctx, tensor.grad)
                if len(tensor._ctx.saved_tensors) == 1:
                    grads = [grads]
                for parent, grad in zip(tensor._ctx.saved_tensors, grads):
                    if isinstance(parent, Tensor) and parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = np.zeros_like(parent.data)
                        parent.grad += grad

    #
    @staticmethod
    def _create_tensor(data, requires_grad=False, dtype=np.float32):
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def _ensure_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def _check_scalar(x):
        assert isinstance(x, (float, int))

    # ----------
    # Creational Ops
    # ----------
    @classmethod
    def kaiming_uniform(cls, shape, gain=math.sqrt(5), mode='fan_in', requires_grad=False, dtype=np.float32):
        data = init.kaiming_uniform(shape, gain=gain, mode=mode)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def kaiming_normal(cls, shape, gain=math.sqrt(5), mode='fan_in', requires_grad=False, dtype=np.float32):
        data = init.kaiming_normal(shape, gain=gain, mode=mode)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def xavier_uniform(cls, shape, gain=1.0, requires_grad=False, dtype=np.float32):
        data = init.xavier_uniform(shape, gain=gain)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def xavier_normal(cls, shape, gain=1.0, requires_grad=False, dtype=np.float32):
        data = init.xavier_normal(shape, gain=gain)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def zeros(cls, shape, requires_grad=False, dtype=np.float32):
        data = init.zeros(shape)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones(cls, shape, requires_grad=False, dtype=np.float32):
        data = init.ones(shape)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def eye(cls, shape, requires_grad=False, dtype=np.float32):
        data = init.eye(shape)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def uniform(cls, shape, a=0.0, b=1.0, requires_grad=False, dtype=np.float32):
        data = init.uniform(shape, a, b)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def normal(cls, shape, mean=0.0, std=1.0, requires_grad=False, dtype=np.float32):
        data = init.normal(shape, mean, std)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype=None, requires_grad=False):
        data = init.arange(start, stop, step, dtype)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True, retstep=False, dtype=None, requires_grad=False):
        data = init.linspace(start, stop, num, endpoint, retstep, dtype)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def logspace(cls, start, stop, num=50, endpoint=True, base=10.0, dtype=None, requires_grad=False):
        data = init.logspace(start, stop, num, endpoint, base, dtype)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def full(cls, shape, fill_value, dtype=None, requires_grad=False):
        data = init.full(shape, fill_value, dtype)
        return cls(data, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def rand(cls, *shape, requires_grad=False, dtype=np.float32):
        data = init.rand(shape)
        return cls(data, requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, requires_grad=False, dtype=np.float32):
        data = init.randn(shape)
        return cls(data, requires_grad=requires_grad)

    @classmethod
    def randint(cls, low, high=None, size=None, requires_grad=False, dtype=np.int32):
        data = init.randint(low, high, size)
        return cls(data, requires_grad=requires_grad)

    # ----------
    # Unnary Ops
    # ----------

    # ----- activations -------
    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function.

        Returns:
            Tensor: Tensor with ReLU applied element-wise.
        """
        return F.ReLU.apply(self)

    def leaky_relu(self, neg_slope=0.01):
        """
        Apply the Leaky Rectified Linear Unit (Leaky ReLU) activation function.

        Args:
            neg_slope (float, optional): Negative slope coefficient. Default is 0.01.

        Returns:
            Tensor: Tensor with Leaky ReLU applied element-wise.

        References:
            Implementation adapted from:
            https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
        """
        return self.relu() - (-neg_slope * self).relu()

    def tanh(self):
        """
        Apply the hyperbolic tangent (tanh) activation function.

        Returns:
            Tensor: Tensor with tanh applied element-wise.
        """
        return F.Tanh.apply(self)

    def sigmoid(self):
        """
        Apply the sigmoid activation function.

        Returns:
            Tensor: Tensor with sigmoid applied element-wise.
        """
        return F.Sigmoid.apply(self)

    def softmax(self, axis=-1):
        """
        Compute the softmax of the tensor along the specified axis.

        The softmax function converts logits to probabilities.
        Uses a stabilization trick by subtracting the maximum value for numerical stability.

        Args:
            axis (int, optional): Axis along which to compute the softmax. Default is -1.

        Returns:
            Tensor: Tensor containing the softmax probabilities.

        References:
            Implementation adapted from:
            https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
        """
        max_tensor = self.max(axis=axis, keepdims=True)
        m = self - max_tensor  # Stabilization
        e = m.exp()
        return e / e.sum(axis=axis, keepdims=True)

    def log_softmax(self, axis=-1):
        """
        Compute the log-softmax of the tensor along the specified axis.

        The log-softmax function is the logarithm of the softmax function.
        Uses a stabilization trick by subtracting the maximum value for numerical stability.

        Args:
            axis (int, optional): Axis along which to compute the log-softmax. Default is -1.

        Returns:
            Tensor: Tensor containing the log-softmax values.

        References:
            Implementation adapted from:
            https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
        """
        max_tensor = self.max(axis=axis, keepdims=True)
        m = self - max_tensor  # Stabilization
        e = m.exp()
        s = e.sum(axis=axis, keepdims=True)
        return m - s.log()

    def swish(self):
        """
        Swish activation function.
        Computes swish as x * sigmoid(x).

        Returns:
            Tensor: Tensor with swish applied element-wise.

        References:
            Implementation adapted from:
            https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py
        """
        return self * self.sigmoid()
    
    def gelu(self):
        """
        Gaussian Error Linear Unit (GELU) activation function.
        Uses an enhanced polynomial approximation for higher accuracy within a tolerance of 0.0001.
        Lower accuracy compared to pytorch but avoids integrals

        Returns:
            Tensor: Tensor with GELU applied element-wise.
        """
        sqrt_2_over_pi = 0.7978845608028654  # Precomputed constant for GELU approximation
        cubic_coefficient = 0.044715  # Coefficient for the cubic term in GELU approximation
        quartic_coefficient = 0.000654  # Coefficient for the quartic term for enhanced accuracy
        
        return 0.5 * self * (1 + (sqrt_2_over_pi * (self + cubic_coefficient * self**3 + quartic_coefficient * self**4)).tanh())


    # ----- standard -------
    def exp(self):
        return F.Exp.apply(self)

    def exp2(self):
        return F.Exp.apply(self * math.log(2))

    def log(self):
        return F.Log.apply(self)

    def log2(self):
        return self.log() / math.log(2)

    def sqrt(self):
        return F.Sqrt.apply(self)

    def rsqrt(self):
        return self.reciprocal().sqrt()

    def sin(self):
        return F.Sin.apply(self)

    def cos(self):
        return ((math.pi / 2) - self).sin()

    def tan(self):
        return self.sin() / self.cos()

    def square(self):
        return self * self

    def abs(self):
        return F.Abs.apply(self)

    def round(self):
        return F.Round.apply(self)

    def ceil(self):
        return F.Ceil.apply(self)

    def floor(self):
        return F.Floor.apply(self)

    def reciprocal(self):
        return F.Reciprocal.apply(self)
    
    def clone(self):
        return F.Copy.apply(self)

    # ----------
    # Binary Ops
    # ----------
    def add(self, other):
        other = Tensor._ensure_tensor(other)
        return F.Add.apply(self, other)

    def __add__(self, other):
        return self.add(other)

    def sub(self, other):
        other = Tensor._ensure_tensor(other)
        return F.Add.apply(self, F.Neg.apply(other))

    def __sub__(self, other):
        return self.sub(other)

    def neg(self):
        return F.Neg.apply(self)

    def __neg__(self):
        return self.neg()

    def mul(self, other):
        other = Tensor._ensure_tensor(other)
        return F.Mul.apply(self, other)

    def __mul__(self, other):
        return self.mul(other)

    def div(self, other):
        other = Tensor._ensure_tensor(other)
        return F.Mul.apply(self, F.Reciprocal.apply(other))

    def __truediv__(self, other):
        return self.div(other)

    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        return F.MatMul.apply(self, other)

    def __matmul__(self, other):
        return self.matmul(other)

    def pow(self, other):
        Tensor._check_scalar(other)
        return F.Pow.apply(self, other)

    def __pow__(self, other):
        return self.pow(other)

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

    def __rtruediv__(self, other):
        return self / other

    # ----------
    # Ternary Ops
    # ----------

    def where(self, condition, y):
        return F.Where.apply(condition, self, y)

    # ----------
    # Reduce Ops
    # ----------

    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return F.Mean.apply(self, axis=axis, keepdims=keepdims)

    def var(self, axis=None, keepdims=False, unbiased=False):
        """Returns the variance of the array elements along the specified axis."""
        ddof = 1 if unbiased else 0
        variance = np.var(self.data, axis=axis, keepdims=keepdims, ddof=ddof)
        return self._create_tensor(variance, requires_grad=False)

    def std(self, axis=None, keepdims=False, unbiased=False):
        """Returns the standard deviation of the array elements along the specified axis."""
        ddof = 1 if unbiased else 0
        std_dev = np.std(self.data, axis=axis, keepdims=keepdims, ddof=ddof)
        return self._create_tensor(std_dev, requires_grad=False)

    def maximum(self, other):
        other = self._ensure_tensor(other)
        return F.Maximum.apply(self, other)

    def minimum(self, other):
        other = self._ensure_tensor(other)
        return F.Minimum.apply(self, other)

    def max(self, axis=None, keepdims=False):
        return F.Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return F.Min.apply(self, axis=axis, keepdims=keepdims)

    def clip(self, min_, max_):
        return self.maximum(min_).minimum(max_)

    def flatten(self, start_dim, end_dim=-1):
        shape = self.shape
        if end_dim == -1:
            end_dim = len(shape) - 1
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim + 1:]
        return self.reshape(*new_shape)

    # ------------
    # Movement Ops
    # ------------

    def view(self, *shape):
        # alias for reshape
        return self.reshape(*shape)

    def reshape(self, *shape):
        return F.Reshape.apply(self, shape)

    def transpose(self, axes=None):
        return F.Transpose.apply(self, axes)

    # def pad(self, pad_width, contant_values=0):
    #     pass

    # def shrink(self, shrink_dims):
    #     pass


    def squeeze(self, axis=None):
        shape = list(self.shape)
        if axis is None:
            new_shape = [dim for dim in shape if dim != 1]
        else:
            if shape[axis] != 1:
                raise ValueError("Cannot squeeze axis with size not equal to 1")
            new_shape = shape[:axis] + shape[axis + 1:]
        return self.reshape(*new_shape)

    def unsqueeze(self, axis):
        shape = list(self.shape)
        new_shape = shape[:axis] + [1] + shape[axis:]
        return self.reshape(*new_shape)

    def __getitem__(self, index):
        return F.Slice.apply(self, index)

    def __setitem__(self, index, value):
        # Check if the tensor is a view of a leaf tensor that requires gradients
        if self.requires_grad and self.grad_fn is None and self._ctx is not None:
            raise RuntimeError(
                "A view of a leaf Tensor that requires grad is being used in an in-place operation."
            )

        # Check if the tensor has been cloned
        if not self.grad_fn or "CopyBackward" not in self.grad_fn:
            raise RuntimeError("Tensor must be cloned before using setitem operation.")

        value = self._ensure_tensor(value)

        # Apply SetItem operation
        original_ctx = self._ctx
        result = F.SetItem.apply(self, index, value)
        self.data = result.data

        # Modify directly
        self._ctx = result._ctx

        # Maintain history
        self._ctx.saved_tensors[0] = original_ctx.saved_tensors[0]

        self._op = result._op
        self.grad_fn = result.grad_fn

    # ----------
    # Utility Ops
    # ----------
    def argmax(self, axis=None):
        """Returns the indices of the maximum values along an axis."""
        indices = np.argmax(self.data, axis=axis)
        return self._create_tensor(indices, requires_grad=False, dtype=np.int64)

    def argmin(self, axis=None):
        """Returns the indices of the minimum values along an axis."""
        indices = np.argmin(self.data, axis=axis)
        return self._create_tensor(indices, requires_grad=False, dtype=np.int64)


    def __len__(self):
        return len(self.data)

    # def __eq__(self, other):
    #     pass

    def __hash__(self):
        return id(self)

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
        info = f"tensor({self.data})"
        if self.requires_grad:
            info += f", requires_grad={self.requires_grad}"
        if self.grad_fn:
            info += f", grad_fn={self.grad_fn}"
        return info
