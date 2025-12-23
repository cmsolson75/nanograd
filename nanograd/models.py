from . import nn
from .tensor import Tensor


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        hidden_dim: int = 512,
        activation_func: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        flatten: bool = True,
    ):
        super().__init__()
        self.flatten = flatten

        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "swish": nn.Swish,
            "gelu": nn.GELU,
        }
        activation_cls = activations.get(
            activation_func.lower(), nn.ReLU
        )  # Default to ReLU if activation is not found

        layers = [nn.Linear(input_dim, hidden_dim), activation_cls()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        for _ in range(max(num_hidden_layers - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.flatten:
            x = x.flatten(1)
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            activation,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int = 3,
        hidden_dim: int = 256,
        activation_func: str = "gelu",
        dropout: float = 0.1,
        flatten: bool = True,
    ):
        super().__init__()
        self.flatten = flatten
        activations = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "swish": nn.Swish,
            "gelu": nn.GELU,
        }
        activation_cls = activations.get(activation_func.lower(), nn.GELU)

        blocks = [
            ResidualBlock(hidden_dim, activation_cls(), dropout=dropout)
            for _ in range(depth)
        ]
        self.encoder = nn.Sequential(
            nn.Flatten(1) if flatten else nn.Identity(),
            nn.Linear(input_dim, hidden_dim),
            activation_cls(),
        )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.blocks(x)
        return self.head(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        flatten: bool = True,
    ):
        super().__init__()
        self.flatten = flatten
        self.encoder = nn.Sequential(
            nn.Flatten(1) if flatten else nn.Identity(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):
    """
    Simple VAE for low-dimensional toy datasets.
    Returns reconstruction, mean, and log-variance for KL computation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dim: int = 64,
        flatten: bool = True,
    ):
        super().__init__()
        self.flatten = flatten
        self.encoder = nn.Sequential(
            nn.Flatten(1) if flatten else nn.Identity(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor):
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = (0.5 * logvar).exp()
        eps = Tensor.randn(mu.shape)
        z = mu + std * eps
        recon = self.decoder(z)
        return recon, mu, logvar
