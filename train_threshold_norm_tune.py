import argparse
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from EventDataset import CAL_COUNT, ED_COUNT, WLS_FAST_COUNT, WLS_SLOW_COUNT
from train_2d_cnn import (
    SEED,
    load_events,
    make_detector_tensors,
    split_indices,
    train_pos_weight,
)
from train_hybrid_cnn import make_engineered_features


# What this file does:
# - Runs the 2D CNN or hybrid CNN with configurable normalization and model parameters.
# - Sweeps a validation-set logit threshold after training, then reports test metrics at
#   both threshold 0 and the validation-selected threshold.
# - Optionally runs a small predefined parameter grid with --tune.
#
# When to run:
#   python train_threshold_norm_tune.py <datafile1> [<datafile2> ...]
#   python train_threshold_norm_tune.py --model 2d --norm log_block --epochs 30 <datafile>
#   python train_threshold_norm_tune.py --model hybrid --tune <datafile1> [<datafile2> ...]
#
# Main parameters to change:
# - Command-line flags in parse_args() are the easiest way to change one run.
# - tune_configs() controls the predefined --tune grid.
# - BATCH_SIZE controls DataLoader batch size; EPS protects divide-by-zero ratios.
BATCH_SIZE = 64
EPS = 1e-9


# All per-run hyperparameters are grouped here so single runs and --tune runs use
# the same experiment interface.
@dataclass(frozen=True)
class RunConfig:
    norm: str
    lr: float
    weight_decay: float
    dropout: float
    wls_channels: int
    small_channels: int
    hidden_dim: int
    epochs: int


# Return slices for the four flat feature blocks: WLS fast, WLS slow, edge, calibration.
def block_slices():
    fast_end = WLS_FAST_COUNT
    slow_end = fast_end + WLS_SLOW_COUNT
    edge_end = slow_end + ED_COUNT
    cal_end = edge_end + CAL_COUNT
    return [
        slice(0, fast_end),
        slice(fast_end, slow_end),
        slice(slow_end, edge_end),
        slice(edge_end, cal_end),
    ]


# Standardize each detector block using only training-set mean/std.
def standardize_by_block(features, train_indices):
    normalized = features.clone()
    train_features = features[train_indices]
    for block in block_slices():
        mean = train_features[:, block].mean()
        std = train_features[:, block].std()
        if std < 1e-6:
            std = torch.tensor(1.0, dtype=features.dtype)
        normalized[:, block] = (features[:, block] - mean) / std
    return normalized


# Apply the selected input normalization mode before reshaping features for the CNN.
# Modes:
# - raw: leave input values unchanged.
# - log1p: compress large nonnegative counts with log(1 + x).
# - block: standardize each detector block.
# - log_block: log1p first, then block standardization.
# - event_fraction: divide each event's features by that event's total activity.
def normalize_features(features, train_indices, mode):
    if mode == "raw":
        return features
    if mode == "log1p":
        return torch.log1p(torch.clamp(features, min=0.0))
    if mode == "block":
        return standardize_by_block(features, train_indices)
    if mode == "log_block":
        logged = torch.log1p(torch.clamp(features, min=0.0))
        return standardize_by_block(logged, train_indices)
    if mode == "event_fraction":
        totals = features.sum(dim=1, keepdim=True).clamp_min(EPS)
        return features / totals
    raise ValueError(f"Unknown normalization mode: {mode}")


# Local DataLoader helper so this file can use its own BATCH_SIZE.
def make_loader(dataset, indices, shuffle):
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=shuffle)


# Tunable version of DetectorBackbone from train_2d_cnn.py.
# wls_channels and small_channels control the first convolution width for each branch.
class TunableBackbone(nn.Module):
    def __init__(self, wls_channels, small_channels):
        super().__init__()
        self.output_dim = (wls_channels * 2) + (small_channels * 2)
        self.wls_branch = nn.Sequential(
            nn.Conv2d(4, wls_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(wls_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(wls_channels, wls_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(wls_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.small_branch = nn.Sequential(
            nn.Conv2d(2, small_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(small_channels),
            nn.ReLU(),
            nn.Conv2d(small_channels, small_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(small_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, wls, small):
        wls_features = self.wls_branch(wls).flatten(1)
        small_features = self.small_branch(small).flatten(1)
        return torch.cat([wls_features, small_features], dim=1)


# 2D CNN whose capacity is controlled by RunConfig.
class TunableCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = TunableBackbone(config.wls_channels, config.small_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, wls, small):
        return self.classifier(self.backbone(wls, small)).squeeze(-1)


# Hybrid CNN whose classifier sees both CNN features and engineered scalar features.
class TunableHybridCNN(nn.Module):
    def __init__(self, config, engineered_dim):
        super().__init__()
        self.backbone = TunableBackbone(config.wls_channels, config.small_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_dim + engineered_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, wls, small, engineered):
        cnn_features = self.backbone(wls, small)
        return self.classifier(torch.cat([cnn_features, engineered], dim=1)).squeeze(-1)


# Run inference over a loader and collect raw logits plus labels for threshold analysis.
def collect_logits(model, loader):
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            *inputs, batch_labels = batch
            logits.append(model(*inputs))
            labels.append(batch_labels)
    return torch.cat(logits), torch.cat(labels)


# Count confusion-matrix entries at an arbitrary logit threshold.
def confusion_counts(logits, labels, threshold):
    predicted = (logits > threshold).int()
    labels = labels.int()
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    return tp, fp, fn, tn


# Convert confusion counts into the metric used to choose the best validation threshold.
# recall95 is useful when false negatives are expensive: it only scores thresholds with
# at least 95% recall, then breaks ties by accuracy.
def score_counts(tp, fp, fn, tn, metric):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = 0.5 * (recall + specificity)

    if metric == "accuracy":
        return accuracy
    if metric == "f1":
        return f1
    if metric == "balanced":
        return balanced
    if metric == "recall95":
        return accuracy if recall >= 0.95 else -1.0
    raise ValueError(f"Unknown threshold metric: {metric}")


# Produce all reported metrics from confusion counts.
def summarize_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = 100 * (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced = 100 * 0.5 * (recall + specificity)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# Search evenly spaced thresholds between the min and max validation logits.
# The selected threshold is later applied once to the held-out test set.
def threshold_sweep(logits, labels, metric):
    min_logit = logits.min().item()
    max_logit = logits.max().item()
    thresholds = torch.linspace(min_logit, max_logit, steps=401)
    best_threshold = 0.0
    best_score = float("-inf")
    best_counts = None

    for threshold in thresholds.tolist():
        counts = confusion_counts(logits, labels, threshold)
        score = score_counts(*counts, metric=metric)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_counts = counts

    return best_threshold, best_score, summarize_counts(*best_counts)


# Generic training loop with best-validation-loss checkpointing.
# epochs comes from RunConfig, so change --epochs or tune_configs() rather than editing here.
def train_model(name, model, train_loader, valid_loader, criterion, optimizer, epochs):
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")

    for epoch in range(epochs):
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
    return best_valid_loss


# Build the TensorDataset expected by either model type.
# The hybrid model uses engineered features computed from the original, unnormalized features.
def build_dataset(model_type, features, normalized_features, labels, train_indices):
    wls, small = make_detector_tensors(normalized_features)
    if model_type == "2d":
        return TensorDataset(wls, small, labels), 0

    engineered = make_engineered_features(features, train_indices)
    return TensorDataset(wls, small, engineered, labels), engineered.shape[1]


# Execute one complete train/validation/test experiment for a single RunConfig.
def run_experiment(model_type, features, labels, indices, pos_weight, config, threshold_metric):
    train_indices, valid_indices, test_indices = indices
    normalized_features = normalize_features(features, train_indices, config.norm)
    dataset, engineered_dim = build_dataset(model_type, features, normalized_features, labels, train_indices)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    if model_type == "2d":
        model = TunableCNN(config)
    else:
        model = TunableHybridCNN(config, engineered_dim)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    name = (
        f"{model_type} norm={config.norm} lr={config.lr:g} wd={config.weight_decay:g} "
        f"dropout={config.dropout:g} channels={config.wls_channels}/{config.small_channels} hidden={config.hidden_dim}"
    )

    valid_loss = train_model(name, model, train_loader, valid_loader, criterion, optimizer, config.epochs)
    valid_logits, valid_labels = collect_logits(model, valid_loader)
    test_logits, test_labels = collect_logits(model, test_loader)

    best_threshold, best_threshold_score, valid_metrics = threshold_sweep(
        valid_logits,
        valid_labels,
        threshold_metric,
    )
    zero_test_metrics = summarize_counts(*confusion_counts(test_logits, test_labels, 0.0))
    tuned_test_metrics = summarize_counts(*confusion_counts(test_logits, test_labels, best_threshold))

    return {
        "name": name,
        "config": config,
        "valid_loss": valid_loss,
        "threshold": best_threshold,
        "threshold_score": best_threshold_score,
        "valid_metrics": valid_metrics,
        "zero_test_metrics": zero_test_metrics,
        "tuned_test_metrics": tuned_test_metrics,
    }


# Print validation threshold details and test metrics at threshold 0 and tuned threshold.
def print_result(result):
    zero = result["zero_test_metrics"]
    tuned = result["tuned_test_metrics"]
    valid = result["valid_metrics"]
    print(f"\nResult: {result['name']}")
    print(f"Best validation loss: {result['valid_loss']:.4f}")
    print(
        f"Validation threshold={result['threshold']:.4f}: "
        f"acc={valid['accuracy']:.3f}% bal_acc={valid['balanced_accuracy']:.3f}% "
        f"precision={valid['precision']:.3f} recall={valid['recall']:.3f} f1={valid['f1']:.3f}"
    )
    print(
        f"Test @ threshold 0: acc={zero['accuracy']:.3f}% bal_acc={zero['balanced_accuracy']:.3f}% "
        f"precision={zero['precision']:.3f} recall={zero['recall']:.3f} f1={zero['f1']:.3f}"
    )
    print(f"TP: {zero['tp']}, FP: {zero['fp']}, FN: {zero['fn']}, TN: {zero['tn']}")
    print(
        f"Test @ swept threshold: acc={tuned['accuracy']:.3f}% bal_acc={tuned['balanced_accuracy']:.3f}% "
        f"precision={tuned['precision']:.3f} recall={tuned['recall']:.3f} f1={tuned['f1']:.3f}"
    )
    print(f"TP: {tuned['tp']}, FP: {tuned['fp']}, FN: {tuned['fn']}, TN: {tuned['tn']}")


# Predefined grid for --tune. Add/remove RunConfig entries here to try more combinations.
# Field order: norm, lr, weight_decay, dropout, wls_channels, small_channels, hidden_dim, epochs.
def tune_configs(epochs):
    return [
        RunConfig("raw", 1e-3, 0.0, 0.25, 16, 8, 64, epochs),
        RunConfig("log1p", 1e-3, 1e-4, 0.25, 16, 8, 64, epochs),
        RunConfig("log_block", 1e-3, 1e-4, 0.25, 16, 8, 64, epochs),
        RunConfig("log_block", 3e-4, 1e-4, 0.25, 16, 8, 64, epochs),
        RunConfig("log_block", 1e-3, 1e-4, 0.40, 16, 8, 64, epochs),
        RunConfig("event_fraction", 1e-3, 1e-4, 0.25, 16, 8, 64, epochs),
        RunConfig("log_block", 1e-3, 1e-4, 0.25, 32, 16, 96, epochs),
    ]


# Command-line interface for one-off experiments and --tune sweeps.
# Use these flags instead of editing constants when you only want to change one run.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train pair/non-pair CNNs with input normalization, threshold sweep, and small parameter tuning."
    )
    parser.add_argument("datafiles", nargs="+")
    parser.add_argument("--model", choices=["2d", "hybrid"], default="hybrid")
    parser.add_argument(
        "--norm",
        choices=["raw", "log1p", "block", "log_block", "event_fraction"],
        default="log_block",
    )
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced", "f1", "recall95"], default="balanced")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--wls-channels", type=int, default=16)
    parser.add_argument("--small-channels", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


# Script entry point: load data once, create one fixed split, then run either one config or --tune.
def main():
    args = parse_args()
    torch.manual_seed(SEED)
    features, labels = load_events(args.datafiles)
    train_indices, valid_indices, test_indices = split_indices(len(labels))
    pos_weight = train_pos_weight(labels, train_indices)
    indices = (train_indices, valid_indices, test_indices)

    print(f"Loaded {len(labels)} events")
    print(f"Split: {len(train_indices)} train, {len(valid_indices)} validation, {len(test_indices)} test")
    print(f"Train pos_weight: {pos_weight.item():.3f}")
    print(f"Threshold metric: {args.threshold_metric}\n")

    if args.tune:
        results = []
        for config in tune_configs(args.epochs):
            torch.manual_seed(SEED)
            result = run_experiment(args.model, features, labels, indices, pos_weight, config, args.threshold_metric)
            print_result(result)
            results.append(result)

        best = max(results, key=lambda item: item["threshold_score"])
        print("\nBest validation threshold score:")
        print_result(best)
        return

    config = RunConfig(
        args.norm,
        args.lr,
        args.weight_decay,
        args.dropout,
        args.wls_channels,
        args.small_channels,
        args.hidden_dim,
        args.epochs,
    )
    result = run_experiment(args.model, features, labels, indices, pos_weight, config, args.threshold_metric)
    print_result(result)


if __name__ == "__main__":
    main()
