import numpy as np
import math
# import autograd as Context

class Function:
    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        ctx = Context()
        inputs = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
        output = cls.forward(ctx, *inputs)
        result = Tensor(output) # might need to add requires_grad = something
        result._ctx = ctx
        result._op = cls
        return result
    
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
    
    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = Tensor(np.ones_like(self.data), requires_grad=False)
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

    def add(self, other):
        return F.Add.apply(self, other)
    
    def mul(self, other):
        return F.Mul.apply(self, other)

    def __add__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        return self.mul(other)