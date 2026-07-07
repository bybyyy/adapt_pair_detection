import copy
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from EventDataset import CAL_COUNT, ED_COUNT, EventDataset, WLS_FAST_COUNT, WLS_SLOW_COUNT
from PairEventClassifier import PairEventClassifier


# What this file does:
# - Loads one or more EventDataset files.
# - Splits events into train/validation/test sets with a fixed random seed.
# - Trains the existing MLP baseline and a small 2D CNN on the same split.
# - Prints test accuracy and confusion matrices so the two models can be compared.
#
# When to run:
#   python train_2d_cnn.py <datafile1> [<datafile2> ...]
#
# Main parameters to change:
# - BATCH_SIZE: number of events per optimizer step.
# - EPOCHS: number of full passes over the training split.
# - SEED: controls the train/validation/test split and model initialization.
# - LR: Adam learning rate for both the MLP baseline and the 2D CNN.
BATCH_SIZE = 64
EPOCHS = 25
SEED = 42
LR = 0.001


# Load and concatenate feature/label tensors from all input data files.
def load_events(paths):
    features = []
    labels = []
    for path in paths:
        dataset = EventDataset(path)
        features.append(dataset.features_tensor)
        labels.append(dataset.labels_tensor)
    return torch.cat(features), torch.cat(labels)


# Create an 80/10/10 train/validation/test split using the global SEED.
def split_indices(count):
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(count, generator=generator)
    train_end = int(0.8 * count)
    valid_end = int(0.9 * count)
    return indices[:train_end], indices[train_end:valid_end], indices[valid_end:]


# Wrap a subset of a TensorDataset in a DataLoader.
def make_loader(dataset, indices, shuffle):
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=shuffle)


# Compute BCEWithLogitsLoss pos_weight from the training labels to handle class imbalance.
def train_pos_weight(labels, train_indices):
    train_labels = labels[train_indices]
    positives = train_labels.sum().item()
    negatives = len(train_labels) - positives
    return torch.tensor([negatives / positives if positives else 1.0], dtype=torch.float32)


# Convert the flat EventDataset feature vector into image-like detector tensors for the CNN.
# WLS data becomes shape [events, 4 channels, 4 detectors, 75 samples].
# Edge/calibration data becomes shape [events, 2 channels, 4 detectors, 6 values].
def make_detector_tensors(features):
    fast_end = WLS_FAST_COUNT
    slow_end = fast_end + WLS_SLOW_COUNT
    edge_end = slow_end + ED_COUNT
    cal_end = edge_end + CAL_COUNT

    fast = features[:, :fast_end].reshape(-1, 4, 2, 75)
    slow = features[:, fast_end:slow_end].reshape(-1, 4, 2, 75)
    edge = features[:, slow_end:edge_end].reshape(-1, 4, 2, 3).reshape(-1, 4, 6)
    cal = features[:, edge_end:cal_end].reshape(-1, 4, 2, 3).reshape(-1, 4, 6)

    wls = torch.stack(
        [fast[:, :, 0, :], fast[:, :, 1, :], slow[:, :, 0, :], slow[:, :, 1, :]],
        dim=1,
    )
    small = torch.stack([edge, cal], dim=1)
    return wls, small


# Shared CNN feature extractor for the WLS branch and the smaller edge/calibration branch.
class DetectorBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.wls_branch = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.small_branch = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, wls, small):
        wls_features = self.wls_branch(wls).flatten(1)
        small_features = self.small_branch(small).flatten(1)
        return torch.cat([wls_features, small_features], dim=1)


# Binary classifier that uses only CNN-extracted detector features.
class PairEvent2DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DetectorBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 1),
        )

    def forward(self, wls, small):
        return self.classifier(self.backbone(wls, small)).squeeze(-1)


# Generic training loop with validation-loss model selection.
# To tune training length, change EPOCHS; to tune optimizer behavior, change LR above.
def train_model(name, model, train_loader, valid_loader, criterion, optimizer):
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            *inputs, labels = batch
            optimizer.zero_grad()
            loss = criterion(model(*inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                *inputs, labels = batch
                valid_loss += criterion(model(*inputs), labels).item()

        train_loss /= max(len(train_loader), 1)
        valid_loss /= max(len(valid_loader), 1)
        print(f"{name} epoch {epoch + 1:2d}: loss={train_loss:.4f} val_loss={valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)


# Evaluate a trained binary classifier at logit threshold 0.
def evaluate_model(model, loader):
    model.eval()
    true_positive = true_negative = false_positive = false_negative = 0

    with torch.no_grad():
        for batch in loader:
            *inputs, labels = batch
            predicted = (model(*inputs) > 0).int()
            labels = labels.int()

            true_positive += ((predicted == 1) & (labels == 1)).sum().item()
            true_negative += ((predicted == 0) & (labels == 0)).sum().item()
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = 100 * (true_positive + true_negative) / total if total else 0.0
    return accuracy, true_positive, false_positive, false_negative, true_negative


# Print accuracy plus the confusion matrix in a consistent format.
def print_metrics(name, metrics):
    accuracy, true_positive, false_positive, false_negative, true_negative = metrics
    print(f"\n{name} accuracy: {accuracy:.3f}%")
    print(f"{name} confusion matrix:")
    print(f"TP: {true_positive}, FP: {false_positive}")
    print(f"FN: {false_negative}, TN: {true_negative}")


# Train and evaluate the original fully connected baseline on the flat feature vector.
def train_mlp_baseline(features, labels, train_indices, valid_indices, test_indices, pos_weight):
    dataset = TensorDataset(features, labels)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    model = PairEventClassifier(features.shape[1], pos_weight=pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_model("Original MLP", model, train_loader, valid_loader, criterion, optimizer)
    return evaluate_model(model, test_loader)


# Train and evaluate the 2D CNN using reshaped detector tensors.
def train_2d_cnn(features, labels, train_indices, valid_indices, test_indices, pos_weight):
    wls, small = make_detector_tensors(features)
    dataset = TensorDataset(wls, small, labels)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    model = PairEvent2DCNN()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_model("2D CNN", model, train_loader, valid_loader, criterion, optimizer)
    return evaluate_model(model, test_loader)


# Script entry point: parse input files, build the split, train both models, and compare results.
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_2d_cnn.py <datafile1> [<datafile2> ...]")
        return

    torch.manual_seed(SEED)
    features, labels = load_events(sys.argv[1:])
    train_indices, valid_indices, test_indices = split_indices(len(labels))
    pos_weight = train_pos_weight(labels, train_indices)

    print(f"Loaded {len(labels)} events")
    print(f"Split: {len(train_indices)} train, {len(valid_indices)} validation, {len(test_indices)} test")
    print(f"Train pos_weight: {pos_weight.item():.3f}\n")

    mlp_metrics = train_mlp_baseline(features, labels, train_indices, valid_indices, test_indices, pos_weight)
    cnn_metrics = train_2d_cnn(features, labels, train_indices, valid_indices, test_indices, pos_weight)

    print_metrics("Original MLP", mlp_metrics)
    print_metrics("2D CNN", cnn_metrics)
    print(f"\nAccuracy difference: {cnn_metrics[0] - mlp_metrics[0]:+.3f} percentage points")


if __name__ == "__main__":
    main()
