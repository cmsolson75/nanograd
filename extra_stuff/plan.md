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


Graph basics: for autograd
- Good book on it
    - https://www.maths.ed.ac.uk/~v1ranick/papers/wilsongraph.pdf
- https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8
- https://www.geeksforgeeks.org/mathematics-graph-theory-basics-set-1/#
- Good video on it: https://youtu.be/LFKZLXVO-Dg?si=QFYH6ELvdGx0TOvZ
- DFS: https://youtu.be/PMMc4VsIacU?si=VTQ8Uy-yDDVEyVnE
- Topo Sort: https://youtu.be/eL-KzMXSXXI?si=qReJnp8Dj5x23Jhy




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
- nn refactor
    - module
    - optim
    - losses
- Need to handle general broadcast cases


Why refactor:
- Make it more testable
- Make expansion into Conv, and Attention a more painless addition
- Hopefully make it more readable for people trying to learn.
- Add all magic methods into framework: 


Path:
- Refactor codebase: I am at a good place where I have tested against pytorch and can match it in nano networks.
- This refactor should make it more modulare and easier to extend to other functions
- After refactored: make sure to test all edge cases against torch.
- Be able to explain the full codebase before moving onto MNIST
- Add in train and eval mode
- Add in context managment. with nano.no_grad() etc
- Make sure dataclass is fully working. as well as data loader, play with in Pytorch to understand the implementation requirments. Look for non standard functionality similar to reductions in the loss functions.
- MNIST: Get it training
- Implement a basic MNIST algorithm
- Start setting up API for codebase: so clear out unessasary stuff.