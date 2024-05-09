Tensor Developement

I need to simplify the backward.

## General Archetecture & Milestones

Phase 1: Goal, train a simple demo net, archetecture should be simple and work. Backward pass should work for what you need it to. Everything is minimal, no need for a optimizer class, or a trainer, just do it by hand.

Phase 2: Goal: MNIST MLP
- Setup GitHub
- Tensor: Full Ops implementation with testing
- Optimizer: SGD w Momentum and L2 regularization
- Trainer: Basic Eval Loops

Phase 3: Goal: state_save, adam, adamw, batch_norm, data_loader(mnist, cifar10)

Phase 4: Conv & Pooling

Phase 5: Model Creation
- MLP: Save Model -> MNIST
- AlexNet: MNIST
- ResNet: Cifar10
- Yolo V3: Face tracking using web cam

Module Outline
nn
- __init__.py
- tensor.py
- autograd.py
- operations.py
- optim.py
- trainers.py
- module.py
- layers.py
models
- MNIST_linear
- MNIST_Conv(Alex net with batch_norm)
- ResNet: Cifar10
    - All implementations: look at TinyGrad for ResNet implementation



## Helpful to implement

Debug
- Adding in a graph viz could be helpful for debugging the shit out of operations.

Testing
- Make sure the major ops are tested, could use pytest. If there are no tests things will break more. Need to verify that the backwards pass works as well as all ops.
- Need to make sure layers compute what they need too(linear, conv, etc)

## Ops to implement

unary ops
- relu: done
- sigmoid: done
- tanh: done
- exp: done
- log: done
- sin: done


binary ops
- add: done
- mul: done
- matmul: done
- div: done
- pow: done
- square: done
- subtract: done
- negative: done
- "=="
- ">"
- "<"

movement 
- transpose & T: done
- stride: Need to add
- im2col: Need to add
- pad: done
- shrink: done
- reshape: done
- flatten: done
- squeeze & unsqueeze: Need to add


reduce ops(axis)
- sum: done
- mean: done
- max & min: done

creational ops
- zeros: done
- ones: done
- eye: done
- randn: done
- arange: done
- uniform: done
- gaussian: done
- randint: done
- kaiming_uniform: done

conversion ops
- numpy: this will output the numpy data internally

object ops: for data loader
- indexing(numpy style)
- iterable

Backward and Derivation
- Use Khans Algorithm to speed up topo sort.

General Clenlyness
- Use Typing
- Add in error handling


I need to implement a data loader, this will be a iterator that handles batches for you.


## Extra notes
I need to implement the following

Finalize basic ops for Tensor
SGD: this needs to be implemented
Should be able to train MNIST with this setup.

Data Loader and batching:
- Need to make Tensor Iterable
- Need to make Tensor Indexable

Memory consideration: 
- Adding inplace operations could improve my speed when I move to RESNET and Yolo, I will be stuck to CPU so any efficaincy is appreciated.


With regards to convolutions
- Need to first implement it the explicit way with multiple loops
- Use the im2col method to make this simpler.


Need basic stats operations: 
- mean
- median
- std