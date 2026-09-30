"""
MNIST digit classification with an MLP -- Claude's attempt
==========================================================

Run it:        python claudes_attempt.py      (inside the dl-env conda env)

Same layout as mnist_mlp.py and still a pure MLP (only Linear layers -- no
convolutions). The changes, from biggest effect to smallest:

  1. Data augmentation (Section 3b): every time a training image is used it is
     randomly shifted, rotated and rescaled a little, so the network never sees
     the exact same image twice. This was the single biggest improvement --
     without it the MLP just memorizes the training set (100% train accuracy).
  2. Better optimizer (Section 4): AdamW instead of plain SGD, a "one-cycle"
     learning-rate schedule (warm up, then slowly decay to ~0), and label
     smoothing.
  3. Bigger model with BatchNorm (Section 2).
  4. Ensemble (Section 6): train a few MLPs with different random seeds and
     average their predictions. Each one makes slightly different mistakes, so
     the average is more accurate than any single one.

Fair play: every setting below was picked using a validation split (10k images
held out from the 60k training images). The test set is only used to report
accuracy -- it is never trained on and was never used to choose settings.
"""

import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


# =============================================================================
# 1. SETTINGS  -- change these and re-run
# =============================================================================
RUN_NAME      = "Claude: 2x1024 BN, aug, AdamW, single MLP"  # label in the table
EPOCHS        = 50                        # passes over the training set (per model)
BATCH_SIZE    = 128
LEARNING_RATE = 2e-3                      # PEAK learning rate of the one-cycle schedule
WEIGHT_DECAY  = 0.05                      # AdamW weight decay (shrinks weights a bit each step)
LABEL_SMOOTHING = 0.1                     # target is 90% "correct digit", 10% spread over all
N_MODELS      = 1                         # how many MLPs to train and average (1 = no ensemble)
SEED          = 0                         # model i uses seed SEED + i

# Data augmentation strengths (Section 3b). Set all to 0 to turn it off.
MAX_SHIFT     = 2                         # pixels, in each direction
MAX_ROTATE    = 10                        # degrees
MAX_SCALE     = 0.1                       # 0.1 -> random zoom between 90% and 110%


# =============================================================================
# 2. MODEL  -- add layers and activations here
# =============================================================================
# Each hidden block is Linear -> BatchNorm -> ReLU -> Dropout.
#
# BatchNorm1d rescales each hidden unit to mean 0 / std 1 over the batch. This
# keeps the numbers in a healthy range as they flow through the network, which
# makes training faster and less sensitive to the learning rate.
#
# 1024 units was as good as 2048 (and 3 hidden layers was no better than 2) on
# the validation set, so this is the smallest model that hit the top score.
def build_model():
    Hdim = 1024
    return nn.Sequential(
        nn.Linear(784, Hdim),
        nn.BatchNorm1d(Hdim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(Hdim, Hdim),
        nn.BatchNorm1d(Hdim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(Hdim, 10),
    )


# =============================================================================
# 3. DATA  -- downloads MNIST the first time (~10 MB), then loads it in memory
# =============================================================================
# Use the Apple GPU (mps) or an NVIDIA GPU (cuda) if there is one -- this is
# ~5x faster than the CPU for this script.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_mnist():
    train = datasets.MNIST("data", train=True, download=True)
    test = datasets.MNIST("data", train=False, download=True)

    def prepare(images):
        x = images.float() / 255.0       # pixels 0..255 -> 0..1
        x = (x - 0.1307) / 0.3081        # standardize (MNIST mean / std)
        return x.reshape(-1, 784).to(DEVICE)  # flatten 28x28 -> 784

    return (prepare(train.data), train.targets.to(DEVICE),
            prepare(test.data), test.targets.to(DEVICE))


# =============================================================================
# 3b. DATA AUGMENTATION  -- random small shifts / rotations / zooms
# =============================================================================
# A digit is still the same digit if you nudge it 2 pixels left or tilt it 10
# degrees. By showing the network a freshly distorted copy of each image every
# epoch, we effectively get an endless supply of new training images, and the
# network has to learn the *shape* of each digit instead of memorizing exact
# pixel positions. (An MLP has no built-in notion that neighbouring pixels are
# related, so it benefits from this a lot.)
#
# How it works: for each image we build a 2x3 "affine" matrix that describes
# the rotation + zoom + shift, then PyTorch's affine_grid / grid_sample move
# the pixels accordingly (with smooth interpolation between pixels).
def augment(x):
    """Return a randomly shifted / rotated / rescaled copy of a batch of images."""
    n = len(x)

    def uniform(max_value):              # n random numbers in [-max_value, +max_value]
        return (torch.rand(n, device=x.device) * 2 - 1) * max_value

    angle = uniform(MAX_ROTATE) * math.pi / 180   # degrees -> radians
    zoom = 1 + uniform(MAX_SCALE)
    shift_x = uniform(MAX_SHIFT) * 2 / 28         # grid_sample measures positions
    shift_y = uniform(MAX_SHIFT) * 2 / 28         # from -1 to +1 across the 28 pixels

    cos = torch.cos(angle) / zoom
    sin = torch.sin(angle) / zoom
    matrix = torch.stack([
        torch.stack([cos, -sin, shift_x], dim=1),
        torch.stack([sin,  cos, shift_y], dim=1),
    ], dim=1)                                      # shape (n, 2, 3)

    images = x.reshape(n, 1, 28, 28)
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    moved = F.grid_sample(images, grid, align_corners=False, padding_mode="border")
    return moved.reshape(n, 784)


# =============================================================================
# 4. TRAINING AND EVALUATION
# =============================================================================
def predict(model, x):
    """Class probabilities (softmax of the scores) for every image in x."""
    model.eval()
    with torch.no_grad():
        return model(x).softmax(dim=1)


def accuracy(probs, y):
    """Percent of images whose most-likely class is the correct one."""
    return (probs.argmax(dim=1) == y).float().mean().item() * 100


def train_one_epoch(model, optimizer, scheduler, loss_fn, x, y):
    """One pass over the training set. Returns the average loss."""
    model.train()
    order = torch.randperm(len(x), device=x.device)   # shuffle every epoch
    total_loss = 0.0

    for start in range(0, len(x), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        xb, yb = augment(x[batch]), y[batch]   # a new random distortion every time

        loss = loss_fn(model(xb), yb)    # forward pass
        optimizer.zero_grad()
        loss.backward()                  # backward pass (gradients)
        optimizer.step()                 # update the weights
        scheduler.step()                 # adjust the learning rate (every batch)

        total_loss += loss.item() * len(batch)

    return total_loss / len(x)


def train_model(seed, x_train, y_train, x_test, y_test):
    """Build and fully train one MLP. Returns the trained model."""
    torch.manual_seed(seed)
    model = build_model().to(DEVICE)

    # Label smoothing: instead of asking for 100% confidence in the correct
    # digit, the target is ~90%. This stops the network from becoming
    # over-confident and acts as a mild regularizer.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # AdamW: adapts the step size for every weight separately (so it needs much
    # less learning-rate tuning than SGD), and applies weight decay correctly.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    # One-cycle schedule: the learning rate ramps up from small to
    # LEARNING_RATE over the first 20% of training, then smoothly decays to
    # almost 0. Big steps early explore; tiny steps at the end fine-tune.
    steps_per_epoch = math.ceil(len(x_train) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, pct_start=0.2,
        total_steps=EPOCHS * steps_per_epoch)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, scheduler, loss_fn, x_train, y_train)
        if epoch % 10 == 0 or epoch == EPOCHS:
            test_acc = accuracy(predict(model, x_test), y_test)
            print(f"  epoch {epoch:2d}/{EPOCHS}   loss {loss:.4f}   test acc {test_acc:5.2f}%")
    return model


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
    print(f"{'run':<44}{'params':>10}{'epochs':>8}{'lr':>8}"
          f"{'train %':>10}{'test %':>9}{'sec':>8}")
    print("-" * 97)
    for r in rows:
        print(f"{r['run']:<44}{r['params']:>10}{r['epochs']:>8}{r['lr']:>8}"
              f"{r['train_acc']:>10}{r['test_acc']:>9}{r['seconds']:>8}")


# =============================================================================
# 6. MAIN  -- train N_MODELS MLPs and average their predictions
# =============================================================================
def main():
    x_train, y_train, x_test, y_test = load_mnist()

    print(build_model())
    n_params = sum(p.numel() for p in build_model().parameters())
    print(f"Trainable parameters: {n_params:,} per model x {N_MODELS} models")
    print(f"Device: {DEVICE}\n")

    start = time.time()
    train_probs, test_probs = [], []
    for i in range(N_MODELS):
        print(f"Model {i + 1}/{N_MODELS}")
        model = train_model(SEED + i, x_train, y_train, x_test, y_test)
        train_probs.append(predict(model, x_train))
        test_probs.append(predict(model, x_test))

        # Ensemble = average the probabilities of all models trained so far
        ensemble_acc = accuracy(torch.stack(test_probs).mean(dim=0), y_test)
        print(f"  ensemble of {i + 1} model(s): test acc {ensemble_acc:.2f}%\n")
    seconds = time.time() - start

    if N_MODELS == 1:
        torch.save(model.state_dict(), "final_mlp.pt")   # the final trained model
        print("Saved final model to final_mlp.pt")

    train_acc = accuracy(torch.stack(train_probs).mean(dim=0), y_train)
    test_acc = accuracy(torch.stack(test_probs).mean(dim=0), y_test)
    print(f"Final: train acc {train_acc:.2f}%   test acc {test_acc:.2f}%")

    save_and_show_results({
        "run": RUN_NAME,
        "params": n_params * N_MODELS,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "train_acc": f"{train_acc:.2f}",
        "test_acc": f"{test_acc:.2f}",
        "seconds": f"{seconds:.1f}",
    })


if __name__ == "__main__":
    main()


# =============================================================================
# RESULTS SO FAR  (as of 2026-09-25)
# =============================================================================
# Validation accuracy (trained on 50k training images, scored on the other 10k
# held-out training images -- the test set was NOT used):
#   original mnist_mlp.py settings ............................ 97.24%
#   this file's model, single MLP (5 seeds) ................... 99.36 - 99.44%
#   this file's model, ensemble of 5 MLPs ..................... 99.41%
#
# Test accuracy: not measured yet -- a full run of this file (training on all
# 60k images) reports it at the end and adds it to results.csv.
