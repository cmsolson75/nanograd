# ***** broadcasted elementwise ops *****
  def _broadcast_to(self, shape:Tuple[sint, ...]):
    reshape_arg, _ = _pad_left(self.shape, shape)
    if self.ndim > len(shape) or not all(sh in {s,1} or (s==0 and sh==1) for sh,s in zip(reshape_arg, shape)):
      raise ValueError(f"cannot broadcast tensor with shape={self.shape} to {shape=}")
    return F.Expand.apply(self.reshape(reshape_arg), shape=shape) if shape != self.shape else self

  def _broadcasted(self, y:Union[Tensor, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (float, int, bool, Node)), f"{type(y)=}, {y=}"
      if isinstance(self.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      else: y_dtype = dtypes.from_py(y)
      if isinstance(y, Node): y = Tensor.from_node(y, device=self.device)
      else: y = Tensor(dtypes.as_const(y, y_dtype), self.device, y_dtype, requires_grad=False)

    if match_dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # broadcast
    out_shape = _broadcast_shape(x.shape, y.shape)
    return x._broadcast_to(out_shape), y._broadcast_to(out_shape)

def _pad_left(*shps:Tuple[sint, ...], v=1): return tuple((v,) * (max(len(i_) for i_ in shps) - len(i)) + i for i in shps)
def _broadcast_shape(*shps:Tuple[sint, ...]): return tuple(0 if any(sh_ == 0 for sh_ in sh) else max(sh) for sh in zip(*_pad_left(*shps)))


  def reshape(self, shape, *args) -> Tensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    new_shape = argfix(shape, *args)
    new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)])
    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self



# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if si != so)
    return x.expand(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(ReduceOps.SUM, self.expanded_axis).cast(grad_output.dtype)

class Reshape(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.reshape(self.input_shape)


class Sum(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, axis)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.expand(self.input_shape)

