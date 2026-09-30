"""
CIFAR-10 image classification with a CNN -- in-class playground
===============================================================

Run it:        python cifar_cnn.py

What to edit:  Section 1 (settings) and Section 2 (the model).
               Everything else can stay as it is.

Every run is appended to results.csv, and a comparison table of all runs
is printed at the end, so you can see how each change affects accuracy.
Delete results.csv to start a fresh comparison.
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torchvision import datasets


# =============================================================================
# 1. SETTINGS  -- change these and re-run
# =============================================================================
RUN_NAME      = "2 conv CNN, 10 ep"      # label shown in the comparison table
EPOCHS        = 10                       # passes over the training set
BATCH_SIZE    = 64
LEARNING_RATE = 0.01
SEED          = 0                        # same seed -> fair comparison between runs


# =============================================================================
# 2. MODEL  -- add layers and activations here
# =============================================================================
# Each image is 3x32x32: 3 color channels, 32 pixels tall, 32 pixels wide.
# Output is 10 scores, one per CIFAR-10 class. Rules:
#   - the first Conv2d layer must take 3 input channels
#   - the last Linear layer must give 10 outputs
#   - after a Conv2d, use an activation such as nn.ReLU()
#   - nn.MaxPool2d(2) cuts height and width in half
#   - use nn.Flatten() before the first Linear layer
#   - do NOT put an activation after the last layer (the loss applies softmax)
#
# Starting image size:
#       input:              3 x 32 x 32
#       after MaxPool2d:   32 x 16 x 16
#       after MaxPool2d:   64 x  8 x  8
#       flattened size:    64 * 8 * 8 = 4096
#
# Activations to try: nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.GELU()
#
# Dropout: nn.Dropout(0.1) randomly zeroes 10% of the values during training
# (it is switched off automatically when measuring accuracy). Put it after an
# activation, never after the last layer.
def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# =============================================================================
# 3. DATA  -- downloads CIFAR-10 the first time (~170 MB), then loads it in memory
# =============================================================================
def load_cifar10():
    train = datasets.CIFAR10("data", train=True, download=True)
    test = datasets.CIFAR10("data", train=False, download=True)

    def prepare(images):
        x = torch.from_numpy(images).float() / 255.0
        x = x.permute(0, 3, 1, 2)        # NHWC -> NCHW for PyTorch CNNs

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        return (x - mean) / std          # standardize each color channel

    return (
        prepare(train.data),
        torch.tensor(train.targets),
        prepare(test.data),
        torch.tensor(test.targets),
    )


# =============================================================================
# 4. TRAINING AND EVALUATION
# =============================================================================
def accuracy(model, x, y):
    """Percent of images classified correctly."""
    model.eval()
    with torch.no_grad():
        predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item() * 100


def train_one_epoch(model, optimizer, loss_fn, x, y):
    """One pass over the training set. Returns the average loss."""
    model.train()
    order = torch.randperm(len(x))       # shuffle every epoch
    total_loss = 0.0

    for start in range(0, len(x), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        xb, yb = x[batch], y[batch]

        loss = loss_fn(model(xb), yb)    # forward pass
        optimizer.zero_grad()
        loss.backward()                  # backward pass (gradients)
        optimizer.step()                 # update the weights

        total_loss += loss.item() * len(batch)

    return total_loss / len(x)


# =============================================================================
# 5. COMPARISON TABLE  -- results.csv keeps one row per run
# =============================================================================
RESULTS_FILE = "results.csv"
COLUMNS = ["run", "params", "epochs", "lr", "train_acc", "test_acc", "seconds"]


def save_and_show_results(row):
    is_new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    with open(RESULTS_FILE) as f:
        rows = list(csv.DictReader(f))

    print("\nAll runs so far:")
    print(f"{'run':<36}{'params':>10}{'epochs':>8}{'lr':>8}"
          f"{'train %':>10}{'test %':>9}{'sec':>8}")
    print("-" * 89)
    for r in rows:
        print(f"{r['run']:<36}{r['params']:>10}{r['epochs']:>8}{r['lr']:>8}"
              f"{r['train_acc']:>10}{r['test_acc']:>9}{r['seconds']:>8}")


# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    torch.manual_seed(SEED)

    x_train, y_train, x_test, y_test = load_cifar10()

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Trainable parameters: {n_params:,}\n")

    loss_fn = nn.CrossEntropyLoss()      # softmax + negative log-likelihood
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, loss_fn, x_train, y_train)
        train_acc = accuracy(model, x_train, y_train)
        test_acc = accuracy(model, x_test, y_test)
        print(f"epoch {epoch:2d}/{EPOCHS}   loss {loss:.4f}   "
              f"train acc {train_acc:5.2f}%   test acc {test_acc:5.2f}%")
    seconds = time.time() - start

    save_and_show_results({
        "run": RUN_NAME,
        "params": n_params,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "train_acc": f"{train_acc:.2f}",
        "test_acc": f"{test_acc:.2f}",
        "seconds": f"{seconds:.1f}",
    })


if __name__ == "__main__":
    main()
