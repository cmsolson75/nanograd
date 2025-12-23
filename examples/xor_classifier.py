import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, dataloader, datasets, models, nn


def main():
    # Four point XOR dataset
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 1, 1, 0], dtype=np.int64)
    dataset = datasets.TensorDataset(inputs, targets)
    loader = dataloader.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = models.SimpleMLP(
        input_dim=2,
        output_dim=2,
        hidden_dim=16,
        num_hidden_layers=2,
        activation_func="tanh",
        dropout=0.1,
        flatten=False,
    )
    optimizer = nn.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(300):
        for batch_inputs, batch_targets in loader:
            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 50 == 0 or step == 299:
            with_logits = model(Tensor(inputs))
            preds = with_logits.softmax(axis=1).argmax(axis=1).data
            acc = np.mean(preds == targets)
            print(f"step {step:03d} | loss={loss.item():.4f} | acc={acc:.2f}")

    print("Training complete. Final predictions:", preds)


if __name__ == "__main__":
    main()
