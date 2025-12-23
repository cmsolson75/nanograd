"""
Tiny 1D autoencoder demo on synthetic audio-like signals (mixture of sine waves + noise).
Reconstructs sequences of length 512; prints reconstruction loss during training.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, nn, models


def synth_batch(batch_size: int = 64, seq_len: int = 512, num_tones: int = 3, noise_std: float = 0.05):
    t = np.linspace(0, 1, seq_len, endpoint=False)
    batch = []
    for _ in range(batch_size):
        signal = np.zeros_like(t, dtype=np.float32)
        for _ in range(num_tones):
            freq = np.random.uniform(2, 12)  # low-frequency tones to keep it smooth
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.5, 1.0)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        signal /= num_tones
        noise = np.random.randn(seq_len).astype(np.float32) * noise_std
        batch.append(signal + noise)
    return np.stack(batch).astype(np.float32)


def main():
    np.random.seed(0)
    seq_len = 512
    model = models.AutoEncoder(input_dim=seq_len, latent_dim=48, hidden_dim=192, flatten=False)
    optimizer = nn.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    steps = 1500
    batch_size = 64
    for step in range(steps):
        x_np = synth_batch(batch_size=batch_size, seq_len=seq_len)
        x = Tensor(x_np, requires_grad=True)

        recon = model(x)
        loss = loss_fn(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            print(f"step {step:04d} | recon_loss={loss.item():.4f}")

    print("Done. Export `x` and `recon` to numpy to listen or visualize waveforms.")


if __name__ == "__main__":
    main()
