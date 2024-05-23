Tensor Developement

I need to simplify the backward.

## General Archetecture & Milestones

Phase 1: Goal, train a simple demo net, archetecture should be simple and work. Backward pass should work for what you need it to. Everything is minimal, no need for a optimizer class, or a trainer, just do it by hand.

Phase 2: Sklearn datasets test
- Get all major ops implemented
- Match pytorch performance on Sklearn datasets

Phase 3: Goal: MNIST MLP
- Tensor: Full testing suite against pytorch.
- MNIST Dataset
- Cifar10 Dataset
- Dataloader
- MLP MNIST Training run


Phase 3: Goal: state_save, adam, adamw, batch_norm

Phase 4: Conv & Pooling

Phase 5: Model Creation
- MLP: Save Model -> MNIST
- AlexNet: MNIST
- ResNet: Cifar10
- Yolo V3: Face tracking using web cam



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

Notes

I have been testing against pytorch, it isnt working as intended

I think my grads are overloading, that is confusing to me but it looks like grads are saturating, my learning is stagnating. could also be hitting a local optima, but I should miror pytorches performance but be slower.

Tanh is unstable, need to do some kind of stablization to make sure I dont overload the grad, could clip it.