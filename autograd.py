from tensor import Tensor

class Function:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        ctx = Context()
        ctx.parents = [arg for arg in args if isinstance(arg, Tensor)]
        ctx.saved_tensors = []
        output = cls.forward(ctx, *args)
        output = Tensor(output)
        output._ctx = ctx
        output.grad_fn = cls
        return output

class Context:
    def __init__(self):
        self.parents = []
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)


# Example backward implmeentation in tensor.
"""
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
"""