# NanoGrad

NumPy-first neural network microframework built to understand training mechanics end-to-end.

## Highlights
- Tape-based autograd engine with topological backprop and broadcasting-aware ops.
- PyTorch-style building blocks: Tensor API, `nn.Module`, `Sequential`, Linear layers, activations, `Flatten`, `BatchNorm1d/2d`, `LayerNorm`, and `Dropout`.
- Optimizers: SGD (momentum/Nesterov), Adam/AdamW, RMSProp, Adagrad, and Adadelta.
- Strong default initializers (Kaiming/Xavier/normal/uniform) and utility tensor creators.
- Ready-to-use models: configurable `SimpleMLP`, residual `ResidualMLP`, `AutoEncoder`, and `VAE` for toy latent-variable experiments.
- Demos for quick inspection (`examples/xor_classifier.py`, `examples/spiral_classifier.py`, `examples/audio_autoencoder.py`, `examples/vae_toy.py`).

## Install
```bash
pip install -e .
# or with test extras
pip install -e ".[dev]"
```

## Quickstart
```python
import numpy as np
from nanograd import Tensor, nn, models

# Tiny classifier
model = models.SimpleMLP(input_dim=2, output_dim=2, hidden_dim=32, num_hidden_layers=2, activation_func="tanh", dropout=0.1, flatten=False)
optimizer = nn.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

inputs = Tensor(np.random.rand(32, 2).astype(np.float32))
targets = Tensor(np.random.randint(0, 2, size=(32,)), dtype=np.int64)

for _ in range(200):
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Demos
- `python examples/xor_classifier.py` – solve XOR with a 2-layer MLP.
- `python examples/spiral_classifier.py` – classify a synthetic spiral dataset with a residual MLP.
- `python examples/audio_autoencoder.py` – train a 1D autoencoder on synthetic audio-like signals.
- `python examples/vae_toy.py` – (optional) tiny VAE on a 2D Gaussian mixture with recon/KL logging.

## Testing
```bash
pytest
```
(tests live in `tests/`)

## Project Layout
- `src/nanograd/` – core Tensor/autograd engine, layers, initializers, and ready-made models.
- `examples/` – runnable demonstration scripts.
- `tests` (`test_*.py`) – parity checks against PyTorch for ops and losses.
