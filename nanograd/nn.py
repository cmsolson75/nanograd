import numpy as np
from typing import Iterable, Generator, Optional, Tuple, List
import math

from .tensor import Tensor
from . import functions as F


class Module:
    """
    Base class for all neural network modules.
    """

    def __init__(self) -> None:
        self.training = True

    def train(self) -> "Module":
        """
        Sets the module in training mode and propagates to children.
        """
        self.training = True
        for child in self._children():
            child.train()
        return self

    def eval(self) -> "Module":
        """
        Sets the module in evaluation mode and propagates to children.
        """
        self.training = False
        for child in self._children():
            child.eval()
        return self

    def _children(self):
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield item

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns an iterator over module parameters.

        Yields
        ------
        Tensor
            A parameter tensor of the module.
        """
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor):
                        yield item
                    elif isinstance(item, Module):
                        yield from item.parameters()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Makes the module callable.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        raise NotImplementedError


class Linear(Module):
    """
    Linear layer (fully connected layer).
    """

    def __init__(self, in_dims: int, out_dims: int, bias: bool = True) -> None:
        """
        Initializes the linear layer with the given dimensions.

        Args:
            in_dims (int): Number of input dimensions.
            out_dims (int): Number of output dimensions.
        """
        super().__init__()
        self.weight = Tensor.kaiming_uniform(
            (in_dims, out_dims),
            a=math.sqrt(5),
            mode="fan_in",
            nonlinearity="leaky_relu",
            requires_grad=True,
        )
        self.bias = Tensor.intercept_uniform(
            (1, out_dims), self.weight.data, requires_grad=True
        )
        # if bias:
        #     self.bias = Tensor.zeros((1, out_dims), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the linear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return x.linear(self.weight, self.bias)

    def __repr__(self) -> str:
        return f"Linear({self.weight.shape})"


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        """
        Initializes the Sequential container with a sequence of modules.

        Args:
            *args (Module): Modules to be added in sequence.
        """
        super().__init__()
        self.modules = args

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes the input tensor through each module in sequence.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all modules.
        """
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns an iterator over module parameters.

        Yields
        ------
        Tensor
            A parameter tensor of the module.
        """
        for module in self.modules:
            yield from module.parameters()

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Conv1d(Module):
    pass


class Conv2d(Module):
    pass


class MaxPool1d(Module):
    pass


class MaxPool2d(Module):
    pass


class AvgPool1d(Module):
    pass


class AvgPool2d(Module):
    pass


# ------------------------------


class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(
        self, parameters: Iterable[Tensor], lr: float, weight_decay: float = 0
    ) -> None:
        """
        Initializes the optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
        """
        self.parameters = list(parameters)
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def zero_grad(self, lazy: bool = True) -> None:
        """
        Sets gradients of all parameters to zero.

        Args:
            lazy (bool): If True, sets gradients to None. Otherwise, sets gradients to zero arrays. Default is True.
        """
        for param in self.parameters:
            if param.requires_grad:
                if lazy:
                    param.grad = None
                else:
                    param.grad = np.zeros_like(param.data)

    def step(self) -> None:
        """
        Performs a single optimization step.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum, weight decay, dampening, and Nesterov momentum.
    <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
    ) -> None:
        """
        Initializes the SGD optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            momentum (float): Momentum factor. Default is 0.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            dampening (float): Dampening for momentum. Default is 0.
            nesterov (bool): Enables Nesterov momentum. Default is False.
        """
        super().__init__(parameters, lr, weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.velocity = [None for p in self.parameters if p.requires_grad]

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                if self.momentum != 0:
                    if self.velocity[idx] is None:
                        self.velocity[idx] = grad
                    else:
                        self.velocity[idx] = (
                            self.momentum * self.velocity[idx]
                            + (1 - self.dampening) * grad
                        )

                    if self.nesterov:
                        grad += self.momentum * self.velocity[idx]
                    else:
                        grad = self.velocity[idx]

                param.data -= self.learning_rate * grad


class Adagrad(Optimizer):
    """
    Adagrad Optimizer

    References:
        Pytorch Implementation Logic: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        Paper: https://jmlr.org/papers/v12/duchi11a.html
    """
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        weight_decay: float = 0,
        epsilon: float = 1e-8,
        initial_accumulator_value: float = 0,
    ):
        """
        Initializes the Adagrad optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            epsilon (float): Small constant to avoid division by zero. Default is 1e-8.
            initial_accumulator_value (float): Initial value for the accumulator. Default is 0.
        """
        super().__init__(parameters, lr, weight_decay)
        self.epsilon = epsilon
        self.state_sum = [
            np.full_like(param.data, initial_accumulator_value)
            for param in self.parameters
            if param.requires_grad
        ]

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                self.state_sum[idx] += grad**2
                param.data -= (
                    self.learning_rate / (np.sqrt(self.state_sum[idx]) + self.epsilon)
                ) * grad


class Adadelta(Optimizer):
    """
    Adadelta optimizer
    References:
        Pytorch Implementation Logic: https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta
        Paper: https://arxiv.org/abs/1212.5701
    """
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0,
        epsilon: float = 1e-6,
    ):
        """
        Initializes the Adadelta optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate (initial value, typically kept as 1.0). Default is 1.0.
            rho (float): Decay rate for the moving average of squared gradients. Default is 0.9.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            epsilon (float): Small constant to avoid division by zero. Default is 1e-6.
        """
        super().__init__(parameters, lr, weight_decay)
        self.rho = rho
        self.epsilon = epsilon
        self.squared_avg = [
            np.zeros_like(param.data) for param in self.parameters if param.requires_grad
        ]
        self.accumulate_updates = [
            np.zeros_like(param.data) for param in self.parameters if param.requires_grad
        ]

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                # Accumulate gradient
                self.squared_avg[idx] = self.rho * self.squared_avg[idx] + (1 - self.rho) * grad ** 2

                # Compute update
                update = (np.sqrt(self.accumulate_updates[idx] + self.epsilon) / 
                          np.sqrt(self.squared_avg[idx] + self.epsilon)) * grad

                # Apply update
                param.data -= self.learning_rate * update

                # Accumulate updates
                self.accumulate_updates[idx] = self.rho * self.accumulate_updates[idx] + (1 - self.rho) * update ** 2


class RMSProp(Optimizer):
    """
    Root Mean Squared Propagation Optimizer

    References:
        Pytorch Logic: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        Hinton Lecture Slides: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        weight_decay: float = 0,
        momentum: float = 0,
        epsilon: float = 1e-8,
        centered: bool = False,
    ):
        """
        Initializes the RMSProp optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate. Default is 0.01.
            alpha (float): Smoothing constant. Default is 0.99.
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            momentum (float): Momentum factor. Default is 0.
            epsilon (float): Small constant to avoid division by zero. Default is 1e-8.
            centered (bool): If True, compute the centered RMSProp. Default is False.
        """
        super().__init__(parameters, lr, weight_decay)
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

        self.square_avg = [
            np.zeros_like(param.data) for param in self.parameters if param.requires_grad
        ]
        if self.momentum > 0:
            self.buffer = [
                np.zeros_like(param.data) for param in self.parameters if param.requires_grad
            ]
        if self.centered:
            self.grad_avg = [
                np.zeros_like(param.data) for param in self.parameters if param.requires_grad
            ]

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad

                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                self.square_avg[idx] = self.alpha * self.square_avg[idx] + (1 - self.alpha) * grad ** 2

                if self.centered:
                    self.grad_avg[idx] = self.alpha * self.grad_avg[idx] + (1 - self.alpha) * grad
                    avg = self.square_avg[idx] - self.grad_avg[idx] ** 2
                else:
                    avg = self.square_avg[idx]

                if self.momentum > 0:
                    self.buffer[idx] = self.momentum * self.buffer[idx] + grad / (np.sqrt(avg) + self.epsilon)
                    param.data -= self.learning_rate * self.buffer[idx]
                else:
                    param.data -= self.learning_rate * grad / (np.sqrt(avg) + self.epsilon)



class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        epsilon: float = 1e-8,
    ):
        """
        Initializes the Adam optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate. Default is 0.001.
            betas (tuple): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
            weight_decay (float): Weight decay (L2 penalty). Default is 0.
            epsilon (float): Small constant to avoid division by zero. Default is 1e-8.
        """
        super().__init__(parameters, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.m = [np.zeros_like(param.data) for param in self.parameters if param.requires_grad]
        self.v = [np.zeros_like(param.data) for param in self.parameters if param.requires_grad]
        self.t = 0

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        self.t += 1
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay != 0:
                    grad += self.weight_decay * param.data

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)



class AdamW(Adam):
    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
    ):
        """
        Initializes the AdamW optimizer.

        Args:
            parameters (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate. Default is 0.001.
            betas (tuple): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
            weight_decay (float): Weight decay. Default is 0.01.
            epsilon (float): Small constant to avoid division by zero. Default is 1e-8.
        """
        super().__init__(parameters, lr, betas, weight_decay, epsilon)

    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        self.t += 1
        for idx, param in enumerate(self.parameters):
            if param.grad is not None:
                grad = param.grad

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # Apply weight decay separately
                if self.weight_decay != 0:
                    param.data -= self.learning_rate * self.weight_decay * param.data



# ------------------------------------------


class Loss:
    """
    Base class for all loss functions.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def _reduce(self, loss):
        reduction_methods = {"mean": loss.mean, "sum": loss.sum, "none": lambda: loss}
        return reduction_methods.get(self.reduction, reduction_methods["none"])()

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Computes the loss between predictions and targets.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            Tensor: Computed loss.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError


class NLLLoss(Loss):
    """
    Negative Log Likelihood loss.
    """

    def __init__(self, weight: Tensor = None, reduction: str = "mean"):
        super().__init__(reduction)
        self.weight = weight

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Negative Log Likelihood loss.

        Args:
            y_hat (Tensor): Log probabilities.
            y (Tensor): True labels.

        Returns:
            float: Computed loss.
        """
        N = y_hat.data.shape[0]
        log_probs = y_hat
        nll = -log_probs[np.arange(N), y.data]
        if self.weight is not None:
            nll *= self.weight[y.data]
        return self._reduce(nll)


class MSELoss(Loss):
    """
    Mean Squared Error loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Mean Squared Error loss.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            float: Computed loss.
        """
        loss = (y_hat - y).square()
        return self._reduce(loss)


class BCELoss(Loss):
    """
    Binary Cross-Entropy loss.
    """

    def __init__(self, weight: Tensor = None, reduction: str = "mean") -> None:
        """
        Initializes the BCELoss with the specified reduction method.

        Args:
            weight (Tensor, optional): Manual rescaling weight given to each class.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super().__init__(reduction)
        self.weight = weight

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Computes the Binary Cross-Entropy loss.

        Args:
            y_hat (Tensor): Predicted probabilities.
            y (Tensor): Ground truth labels.

        Returns:
            Tensor: The computed loss.
        """
        epsilon = 1e-6
        y_hat = y_hat.clip(epsilon, 1 - epsilon)
        loss = -(y * y_hat.log() + (1 - y) * (1 - y_hat).log())
        if self.weight is not None:
            loss *= self.weight
        return self._reduce(loss)


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy loss.
    """

    def __init__(self, weight: Tensor = None, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.weight = weight

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Cross-Entropy loss.

        Args:
            y_hat (Tensor): Logits.
            y (Tensor): True labels.

        Returns:
            float: Computed loss.
        """
        log_prob = y_hat.log_softmax(axis=1)
        loss = NLLLoss(self.weight, self.reduction)(log_prob, y)
        return loss


class L1Loss(Loss):
    """
    L1 loss (Mean Absolute Error).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the L1 loss.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            float: Computed loss.
        """
        loss = (y_hat - y).abs()
        return self._reduce(loss)


class HuberLoss(Loss):
    """
    Huber loss.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        """
        Initializes the HuberLoss with the specified delta and reduction method.

        Args:
            delta (float): The threshold at which to change between delta-scaled MAE and MSE.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super().__init__(reduction)
        self.delta = delta

    def __call__(self, y_hat: Tensor, y: Tensor) -> float:
        """
        Computes the Huber loss.

        Args:
            y_hat (Tensor): Predicted values.
            y (Tensor): True values.

        Returns:
            float: Computed loss.
        """
        error = y_hat - y
        abs_error = error.abs()
        quadratic = 0.5 * error.square()
        linear = self.delta * (abs_error - 0.5 * self.delta)
        loss = quadratic.where(abs_error.data <= self.delta, linear)
        return self._reduce(loss)


# ---------------


class BatchNorm1d(Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Tensor.ones((1, num_features), requires_grad=True)
        self.beta = Tensor.zeros((1, num_features), requires_grad=True)
        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            batch_var = ((x - batch_mean).square()).mean(axis=0, keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data

            mean = batch_mean
            var = batch_var
        else:
            mean = Tensor(self.running_mean, requires_grad=False)
            var = Tensor(self.running_var, requires_grad=False)

        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta


class BatchNorm2d(Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Tensor.ones((1, num_features, 1, 1), requires_grad=True)
        self.beta = Tensor.zeros((1, num_features, 1, 1), requires_grad=True)
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            axes = (0, 2, 3)
            batch_mean = x.mean(axis=axes, keepdims=True)
            batch_var = ((x - batch_mean).square()).mean(axis=axes, keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data

            mean = batch_mean
            var = batch_var
        else:
            mean = Tensor(self.running_mean, requires_grad=False)
            var = Tensor(self.running_var, requires_grad=False)

        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.gamma = Tensor.ones(self.normalized_shape, requires_grad=True)
        self.beta = Tensor.zeros(self.normalized_shape, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        axis_start = x.ndims - len(self.normalized_shape)
        axes = tuple(range(axis_start, x.ndims))
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean).square()).mean(axis=axes, keepdims=True)
        normalized = (x - mean) / (var + self.eps).sqrt()

        broadcast_shape = (1,) * axis_start + self.normalized_shape
        gamma = self.gamma.reshape(*broadcast_shape)
        beta = self.beta.reshape(*broadcast_shape)
        return normalized * gamma + beta


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1).")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        mask = Tensor(
            np.random.binomial(1, keep_prob, size=x.shape).astype(x.data.dtype),
            requires_grad=False,
        )
        return x * mask / keep_prob


# ----------------- For Sequential


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def forward(self, x):
        return x.leaky_relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    
class Swish(Module):
    def forward(self, x):
        return x.swish()

class GELU(Module):
    def forward(self, x):
        return x.gelu()
