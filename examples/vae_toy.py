import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, nn, models


def sample_gaussians(batch_size: int = 128):
    centers = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )
    idx = np.random.randint(0, len(centers), size=batch_size)
    noise = 0.1 * np.random.randn(batch_size, 2).astype(np.float32)
    return centers[idx] + noise


def vae_loss(x, recon, mu, logvar, beta: float = 1.0):
    recon_loss = (recon - x).square().mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl.mean()
    return recon_loss + beta * kl, recon_loss, kl


def main():
    np.random.seed(0)
    model = models.VAE(input_dim=2, latent_dim=2, hidden_dim=64, flatten=False)
    optimizer = nn.Adam(model.parameters(), lr=3e-3)

    steps = 2000
    for step in range(steps):
        batch = Tensor(sample_gaussians(256))
        recon, mu, logvar = model(batch)

        beta = min(0.5, 0.5 * step / 400)  # KL warmup for stability
        loss, recon_loss, kl_loss = vae_loss(batch, recon, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            print(
                f"step {step:04d} | loss={loss.item():.4f} "
                f"(recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}, beta={beta:.3f})"
            )

    print("Done. You can visualize reconstructions or latent samples by exporting `recon`/`mu` to numpy.")


if __name__ == "__main__":
    main()
