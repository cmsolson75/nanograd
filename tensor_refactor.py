import numpy as np
import math

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
        return result

# import at this point to stop circular imports
import functions as F


class Context:
    def __init__(self):
        self.parents = []
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = (data if isinstance(data, np.ndarray) else np.array(data)).astype(
            dtype=dtype
        )
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None
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
                    print(grads.shape)
                    grads = [grads]
                for parent, grad in zip(tensor._ctx.saved_tensors, grads):
                    if isinstance(parent, Tensor) and parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = np.zeros_like(parent.data)
                        parent.grad += grad

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
    def kaiming_uniform(cls, in_dims, out_dims, gain: float = math.sqrt(5), **kwargs):
        pass

    @classmethod
    def kaiming_normal(self):
        pass
    
    @classmethod
    def xavier_uniform(self):
        pass

    @classmethod
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

    

    # ----------
    # Unnary Ops
    # ----------

    # ----- activations -------
    def relu(self):
        return F.ReLU.apply(self)
    
    def leaky_relu(self, neg_slope=0.01):
        pass

    def tanh(self):
        return F.Tanh.apply(self)

    def sigmoid(self):
        return F.Sigmoid.apply(self)
    
    def softmax(self, axis=-1):
        pass

    def log_softmax(self, axis=-1):
        pass

    def swish(self):
        pass

    def gelu(self):
        pass

    # ----- standard -------
    def transpose(self):
        pass

    def exp(self):
        pass

    def exp2(self):
        pass

    def log(self):
        pass

    def log2(self):
        pass

    def sqrt(self):
        pass

    def rsqrt(self):
        pass

    def sin(self):
        pass

    def cos(self):
        pass

    def tan(self):
        pass

    def square(self):
        return self * self

    def abs(self):
        pass

    def round(self):
        pass

    def ceil(self):
        pass

    def floor(self):
        pass

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
    
    # ----------
    # Ternary Ops
    # ----------

    def where(self, condition, y):
        pass

    # ----------
    # Reduce Ops
    # ----------
    
    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return F.Mean.apply(self, axis=axis, keepdims=keepdims)

    def maximum(self):
        pass

    def minimum(self):
        pass

    def max(self):
        pass

    def min(self):
        pass

    def clip(self):
        pass

    def flatten(self, start_dim, end_dim=-1):
        pass

    # ------------
    # Movement Ops
    # ------------
    
    def view(self, *shape):
        # alias for reshape
        pass

    def reshape(self, *shape):
        pass

    def pad(self, pad_width, contant_values=0):
        pass

    def shrink(self, shrink_dims):
        pass

    def squeeze(self, axis=None):
        pass

    def unsqueeze(self, axis):
        pass

    def __getitem__(self, index):
        pass

    def __setitem__(self, morestuff):
        pass

    # ----------
    # Utility Ops
    # ----------
    def argmax(self, axis):
        pass

    def argmin(self, axis):
        pass

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
        return f"tensor({self.data})"
