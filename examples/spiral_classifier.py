import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanograd import Tensor, nn, models


def make_spiral(points_per_class: int = 100, num_classes: int = 3, noise: float = 0.2):
    total_points = points_per_class * num_classes
    data = np.zeros((total_points, 2), dtype=np.float32)
    labels = np.zeros(total_points, dtype=np.int64)

    for class_idx in range(num_classes):
        ix = slice(class_idx * points_per_class, (class_idx + 1) * points_per_class)
        radius = np.linspace(0.0, 1, points_per_class)
        theta = np.linspace(class_idx * 4, (class_idx + 1) * 4, points_per_class)
        theta += np.random.randn(points_per_class) * noise

        data[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        labels[ix] = class_idx
    return data, labels


def main():
    inputs, targets = make_spiral(points_per_class=150, num_classes=3, noise=0.2)

    model = models.ResidualMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=64,
        depth=4,
        activation_func="relu",
        dropout=0.1,
        flatten=False,
    )
    optimizer = nn.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(800):
        logits = model(Tensor(inputs))
        loss = loss_fn(logits, Tensor(targets, dtype=np.int64))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 799:
            probs = logits.softmax(axis=1)
            preds = probs.argmax(axis=1).data
            acc = np.mean(preds == targets)
            print(f"step {step:03d} | loss={loss.item():.4f} | acc={acc:.3f}")

    print("Done! Try plotting preds vs. inputs for a quick decision boundary visualization.")


if __name__ == "__main__":
    main()
