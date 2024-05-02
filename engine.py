import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _prev=()):
        self.data = (data if isinstance(data, np.ndarray) 
                     else np.array(data)).astype(np.float32)
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_prev)
        self.grad_fn = str() # for debug
        self.requires_grad = requires_grad

    # --------------
    # Creational ops
    # --------------

    @classmethod
    def kaiming_uniform(cls, in_dims, out_dims, **kwargs):
        limit = np.sqrt(6 / in_dims)
        data = np.random.uniform(-limit, limit, (in_dims, out_dims))
        return cls(data, **kwargs)
    
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
    def arange(cls, start, stop, step, **kwargs):
        return cls(np.arange(start, stop, step), **kwargs)
    
    @property
    def T(self):
        out = Tensor(self.data.T, _prev=(self,))

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

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndims(self):
        return len(self.shape)
    
    # ----------
    # Binary Ops
    # ----------

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0], f"{self.shape}, {other.shape}: Incompatible shapes for matrix multiplication."
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
        out = Tensor(self.data + other.data, _prev=(self, other))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_self, _ = np.broadcast_arrays(out.grad, self.data)
                axes_to_reduce = tuple(i for i in range(grad_self.ndim) if self.data.shape[i] != grad_self.shape[i])
                self.grad += np.sum(grad_self, axis=axes_to_reduce, keepdims=True)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                _, grad_other = np.broadcast_arrays(out.grad, other.data)
                axes_to_reduce = tuple(i for i in range(grad_other.ndim) if other.data.shape[i] != grad_other.shape[i])
                other.grad += np.sum(grad_other, axis=axes_to_reduce, keepdims=True)

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
        out = Tensor(np.power(self.data, other), _prev=(self, ))

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
    # unary ops
    # ---------
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
        n = self.data
        t = (np.exp(2*n) - 1)/(np.exp(2*n)+1) # Calc tanh
        out = Tensor(t, _prev=(self, ))
        
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
        s = 1/(1 + np.exp(-x))
        out = Tensor(s, _prev=(self, ))

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
        
    def exp(self):
        out = Tensor(np.exp(self.data), _prev=(self, ))
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
        out = Tensor(np.log(self.data), _prev=(self, ))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1/self.data) * out.grad

        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "LogBackward"
        return out
    
    def sin(self):
        out = Tensor(np.sin(self.data), _prev=(self, ))
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
    
    # ----------
    # reduce ops
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
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), 
                     _prev=(self, ))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                # distribute grad with scale 1/n
                grad = np.full(self.data.shape, 1/self.data.size)
                self.grad += grad * out.grad
        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MeanBackward"
        
        return out
    
    def max(self, axis=None, keepdims=False):
        max_value = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_value, _prev=(self, ))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_mask = (self.data == np.expand_dims(max_value, axis=axis))
                self.grad += grad_mask * out.grad
        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MaxBackward"
        return out

    def min(self, axis=None, keepdims=False):
        min_value = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(min_value, _prev=(self, ))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad_mask = (self.data == np.expand_dims(min_value, axis=axis))
                self.grad += grad_mask * out.grad
        if self.requires_grad:
            out.requires_grad = True
            out._backward = _backward
            out.grad_fn = "MinBackward"
        return out
    
    # ------------
    # Movement Ops
    # ------------

    def reshape(self, *shape):
        reshaped_data = self.data.reshape(*shape)
        out = Tensor(reshaped_data, _prev=(self, ))

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
    
    # ----------
    # Utility Ops
    # ----------

    def argmax(self, axis=None):
        """ Returns the indices of the maximum values along an axis. """
        indices = np.argmax(self.data, axis=axis)
        return indices
    
    def argmin(self, axis=None):
        """ Returns the indices of the minimum values along an axis. """
        indices = np.argmin(self.data, axis=axis)
        return indices
    

    def __repr__(self):
        return f"{self.data}"
    
    # ---------
    # Backwards
    # ---------

    def _topological_sort(self):
        # Update to khan algorithm
        # Review Graph Theory
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