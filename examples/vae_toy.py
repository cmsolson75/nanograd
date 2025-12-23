import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, dataloader, datasets, models, nn


def vae_loss(x, recon, mu, logvar, beta: float = 1.0):
    recon_loss = (recon - x).square().mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl.mean()
    return recon_loss + beta * kl, recon_loss, kl


def main():
    np.random.seed(0)
    centers = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )
    dataset = datasets.GaussianMixtureDataset(centers=centers, noise=0.08, length=40000, seed=7)
    loader = dataloader.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    model = models.VAE(input_dim=2, latent_dim=2, hidden_dim=128, flatten=False)
    optimizer = nn.Adam(model.parameters(), lr=2e-3)

    num_epochs = 35
    warmup_steps = 800  # gradual KL ramp for better recon quality
    beta_cap = 0.35
    log_interval = 400
    ema_loss = ema_recon = ema_kl = None
    ema_alpha = 0.1
    step = 0
    for epoch in range(num_epochs):
        for batch in loader:
            recon, mu, logvar = model(batch)

            beta = min(beta_cap, beta_cap * step / warmup_steps)
            loss, recon_loss, kl_loss = vae_loss(batch, recon, mu, logvar, beta=beta)

            # lightweight EMA to smooth noisy minibatch reporting
            def _ema(prev, new):
                return new if prev is None else (1 - ema_alpha) * prev + ema_alpha * new

            ema_loss = _ema(ema_loss, loss.item())
            ema_recon = _ema(ema_recon, recon_loss.item())
            ema_kl = _ema(ema_kl, kl_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_interval == 0:
                print(
                    f"step {step:05d} | loss={ema_loss:.4f} "
                    f"(recon={ema_recon:.4f}, kl={ema_kl:.4f}, beta={beta:.3f})"
                )

        print(
            f"epoch {epoch+1:02d}/{num_epochs} | "
            f"loss={ema_loss:.4f} recon={ema_recon:.4f} kl={ema_kl:.4f} beta={beta:.3f}"
        )

    print("Done. You can visualize reconstructions or latent samples by exporting `recon`/`mu` to numpy.")


if __name__ == "__main__":
    main()
