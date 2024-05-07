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
- square: 
- subtract: done
- negative: done
- "=="
- ">"
- "<"

movement 
- transpose & T: done
- stride
- pad: done
- shrink: done
- reshape: done
- flatten: done
- squeeze & alternate


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

## Code Notes

Not doing a recursive topo due to recursion limits, use a deque

Topological sort: Look this up
This is called Khans Algorithm: look it up

```
def _topological_sort(self):
        # Calculate in-degrees of each node
        in_degree = {self: 0}
        queue = deque([self])  # Start with the final node and explore backwards

        while queue:
            node = queue.popleft()
            for pred in node._prev:
                if pred not in in_degree:
                    in_degree[pred] = 0
                in_degree[pred] += 1
                if pred not in node._next:
                    pred._next.add(node)  # Build the next relationship
                    queue.append(pred)

        # Start with nodes with no incoming edges (in_degree 0)
        zero_in_degree_queue = deque([n for n in in_degree if in_degree[n] == 0])
        topo_sorted = []

        while zero_in_degree_queue:
            node = zero_in_degree_queue.popleft()
            topo_sorted.append(node)
            for next_node in node._next:
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    zero_in_degree_queue.append(next_node)

        if len(topo_sorted) != len(in_degree):
            raise Exception("Graph has at least one cycle, which is not allowed for topological sort")

        return topo_sorted

    def _compute_gradients(self, topo_sorted):
        self.grad = np.ones_like(self.data)  # Initialize the gradient for the starting node
        for node in reversed(topo_sorted):
            node._backward()

    def backward(self):
        topo_sorted = self._topological_sort()
        self._compute_gradients(topo_sorted)
```