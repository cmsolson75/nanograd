What the plan is

Optimizers
- General Introduction: https://arxiv.org/pdf/1609.04747
- SGD: Nestorov, Momentum, L2, and Damp: 
    - https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
- RMSProp
    - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- AdaGrad:
    - https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
- Adam
    - https://arxiv.org/pdf/1412.6980
- AdamW
    - https://arxiv.org/pdf/1711.05101

Normalization
- Batch Normalization: 
    - https://arxiv.org/abs/1502.03167
- Layer Normalization:
    - https://arxiv.org/abs/1607.06450
- Dropout
    - https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf




TODO
- Need to get full test coverage
- Need full operator coverage with python
    - Match Numpy/Torch

TypeError: '<=' not supported between instances of 'Tensor' and 'float'


Need to add in model.train & model.test as well as the context manager


Refactor of Tensor Class
- Break up into
    - autograd.py
        - Function: base class for all ops
        - Context: this is the computational graph
    - functions.py: has all the ops using the Function base implementation
    - tensor.py: general operations.
- Need to handle general broadcast cases


Why refactor:
- Make it more testable
- Make expansion into Conv, and Attention a more painless addition
- Hopefully make it more readable for people trying to learn.
- Add all magic methods into framework: 


Full functionality
- Right hand ops
    - __radd__: Right-hand addition.
    - __rmul__: Right-hand multiplication.
    - __rsub__: Right-hand subtraction.
    - __rtruediv__: Right-hand true division.
- Comparison
    - __lt__: Less than.
    - __le__: Less than or equal to.
    - __gt__: Greater than.
    - __ge__: Greater than or equal to.
- Implement base operators that are separate from the python method: 
    - Example: add(): this would have the add functionality, then you also have __add__(self, a, b): return add(a, b) I think this might help implementation details, and allow different calls of objects.


Path:
- Refactor codebase: I am at a good place where I have tested against pytorch and can match it in nano networks.
- This refactor should make it more modulare and easier to extend to other functions
- After refactored: make sure to test all edge cases against torch.
- Be able to explain the full codebase before moving onto MNIST
- Add in train and eval mode
- Add in context managment. with nano.no_grad() etc