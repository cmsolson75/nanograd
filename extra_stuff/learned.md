This document is for things I am learning


In backprop, the grad moves through the chain rule, but there is more to it. This is an example

next_grad = local_grad * global_grad

There is a better way to think about this though and that comes from learning about sum and reduction ops. In these ops you can think of them as a many to one relationship, what happens when you reverse this, you have a one to many. In the case of the sum, you have to distribute the gradient to all of the many nodes. This gave me the idea of distribution gradients, what is the local grad other than a distribution gradient. Sometimes this distribution gradient is essentially a pass through, but the dims of this determing how many notes get gradient, so even a simple pass through is essential to getting gradient backwards. The most important cases is when the forward data is mixed into the gradient, this can be seen in the mul operation.

Way of thinking about it.
out_grads = distribution_grad * global_grad