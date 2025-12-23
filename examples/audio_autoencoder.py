"""
Tiny 1D autoencoder demo on synthetic audio-like signals (mixture of sine waves + noise).
Reconstructs sequences of length 512; prints reconstruction loss during training.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, dataloader, datasets, models, nn


class SineNoiseDataset(datasets.Dataset):
    """
    Synthetic 1D signals: mixture of sine waves plus Gaussian noise.
    """

    def __init__(self, length=4000, seq_len=512, num_tones=3, noise_std=0.05, seed=0):
        self.length = length
        self.seq_len = seq_len
        self.num_tones = num_tones
        self.noise_std = noise_std
        self.seed = seed
        self.t = np.linspace(0, 1, seq_len, endpoint=False).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        signal = np.zeros_like(self.t, dtype=np.float32)
        for _ in range(self.num_tones):
            freq = rng.uniform(2, 12)  # low-frequency tones to keep it smooth
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.0)
            signal += amp * np.sin(2 * np.pi * freq * self.t + phase)
        signal /= self.num_tones
        noise = rng.normal(0, self.noise_std, size=self.seq_len).astype(np.float32)
        sample = signal + noise
        x = Tensor(sample, dtype=np.float32)
        target = Tensor(sample.copy(), dtype=np.float32)
        return x, target


def main():
    np.random.seed(0)
    seq_len = 512
    dataset = SineNoiseDataset(length=4000, seq_len=seq_len)
    loader = dataloader.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    model = models.AutoEncoder(input_dim=seq_len, latent_dim=48, hidden_dim=192, flatten=False)
    optimizer = nn.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 25 
    step = 0
    for epoch in range(epochs):
        for batch_inputs, batch_targets in loader:
            recon = model(batch_inputs)
            loss = loss_fn(recon, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        print(f"epoch {epoch+1:02d}/{epochs} complete | latest recon_loss={loss.item():.4f}")

    print("Done. Export `x` and `recon` to numpy to listen or visualize waveforms.")


if __name__ == "__main__":
    main()
