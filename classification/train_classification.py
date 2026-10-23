"""Train the patch classification model (vessel vs. background) on one perturbation of the
patch dataset. Run `extract_patches.py` first.

Example:
    # Single run: dataset D_{-i,-t} with f = 100% and seed 1
    python train_classification.py --fraction 1.0 --perturbation it --seed 1

    # All runs of the paper (4 perturbations x 3 fractions x 5 seeds)
    python train_classification.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import ResNetClassifier
from perturbations import PERTURBATIONS, normalize, shuffle_pixels

SEEDS = [1, 2, 10, 1998, 2907]
FRACTIONS = [1.0, 0.75, 0.5]


class PatchDataset(Dataset):

    def __init__(self, patches, labels, seed=2, shuffle=False, normalize=False):
        self.patches = patches
        self.labels = labels
        self.seed = seed
        self.shuffle = shuffle
        self.normalize = normalize

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx]
        if self.shuffle:
            img = shuffle_pixels(img, self.seed)
        if self.normalize:
            img = normalize(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img, self.labels[idx]


def split_dataset(vessel_patches, background_patches, train_frac=0.8, shuffle=False, normalize=False, seed=2):
    """Balance the classes and randomly split the patches into train and test sets."""

    np.random.seed(seed)

    num_files = min(len(vessel_patches), len(background_patches))
    num_train = int(num_files*train_frac)

    indices = np.arange(0, num_files)
    idx_train = np.random.choice(indices, size=num_train, replace=False)
    idx_test = np.setdiff1d(indices, idx_train)

    def make(idx):
        patches = np.concatenate((vessel_patches[idx], background_patches[idx]))
        labels = [1]*len(idx) + [0]*len(idx)
        return PatchDataset(patches, labels, seed, shuffle, normalize)

    return make(idx_train), make(idx_test)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one epoch. Trains the model if `optimizer` is given. Returns (accuracy, loss)."""

    model.train(optimizer is not None)
    total, correct, total_loss = 0, 0, 0.
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predicted = outputs.detach().argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()*labels.size(0)

    return 100*correct/total, total_loss/total


def train(model, dataset_train, dataset_test, batch_size=9, num_epochs=300, lr=0.001, device="cpu"):
    """Train the model and keep the epoch having the smallest test loss.

    The reported accuracy is the test accuracy at that epoch (`acc_test[-1]` of the
    returned history).
    """

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"acc_train": [], "acc_test": [], "loss_train": [], "loss_test": []}
    best = {}
    for epoch in range(num_epochs):
        acc, loss = run_epoch(model, train_loader, criterion, device, optimizer)
        history["acc_train"].append(acc)
        history["loss_train"].append(loss)

        acc, loss = run_epoch(model, test_loader, criterion, device)
        history["acc_test"].append(acc)
        history["loss_test"].append(loss)

        if loss == min(history["loss_test"]):
            best = {
                "epoch": epoch,
                "accuracy": acc,
                "loss": loss,
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "history": {k: v[:] for k, v in history.items()},
            }

    return best


def run(args, fraction, perturbation, seed, device):

    patches_dir = Path(args.patches_dir)
    vessel = np.load(patches_dir / f"vessel_f{int(fraction*100)}.npz")["patches"]
    background = np.load(patches_dir / "background.npz")["patches"]

    shuffle, norm = PERTURBATIONS[perturbation]
    dataset_train, dataset_test = split_dataset(vessel, background, shuffle=shuffle, normalize=norm, seed=seed)

    torch.manual_seed(seed)
    model = ResNetClassifier([2], [64])
    best = train(model, dataset_train, dataset_test, args.batch_size, args.epochs, args.lr, device)

    name = f"{perturbation}_f{int(fraction*100)}_seed{seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        torch.save(best["model_state"], output_dir / f"{name}.pt")
    result = {"perturbation": perturbation, "fraction": fraction, "seed": seed,
              "epoch": best["epoch"], "accuracy": best["accuracy"], "loss": best["loss"],
              "history": best["history"]}
    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"{name}: accuracy {best['accuracy']:.2f}% (epoch {best['epoch']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches_dir", default="patches")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--fraction", type=float, default=1.0, choices=FRACTIONS, help="Fraction f of vessel pixels")
    parser.add_argument("--perturbation", default="none", choices=list(PERTURBATIONS),
                        help="none: D, i: D_{-i}, t: D_{-t}, it: D_{-i,-t}")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the train/test split")
    parser.add_argument("--all", action="store_true", help="Run all configurations used in the paper")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_model", action="store_true", help="Also save the weights of the best epoch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.all:
        configs = [(f, p, s) for f in FRACTIONS for p in PERTURBATIONS for s in SEEDS]
    else:
        configs = [(args.fraction, args.perturbation, args.seed)]

    for fraction, perturbation, seed in configs:
        run(args, fraction, perturbation, seed, args.device)


if __name__ == "__main__":
    main()
