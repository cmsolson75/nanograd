import nn
import Tensor


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        hidden_dim: int = 512,
        activation_func: str = "relu",
        flatten: bool = True,
    ):
        super().__init__()
        self.flatten = flatten

        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "swish": nn.Swish(),
            "gelu": nn.GELU(),
        }
        activation_module = activations.get(
            activation_func.lower(), nn.ReLU()
        )  # Default to ReLU if activation is not found

        layers = [nn.Linear(input_dim, hidden_dim), activation_module]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_module])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.flatten:
            x = x.flatten(1)
        return self.model(x)
