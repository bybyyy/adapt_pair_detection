import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from EventDataset import CAL_COUNT, ED_COUNT, WLS_FAST_COUNT, WLS_SLOW_COUNT
from train_2d_cnn import (
    LR,
    SEED,
    DetectorBackbone,
    evaluate_model,
    load_events,
    make_detector_tensors,
    make_loader,
    print_metrics,
    split_indices,
    train_mlp_baseline,
    train_model,
    train_pos_weight,
)


class PairEventHybridCNN(nn.Module):
    def __init__(self, engineered_dim):
        super().__init__()
        self.backbone = DetectorBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(48 + engineered_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 1),
        )

    def forward(self, wls, small, engineered):
        cnn_features = self.backbone(wls, small)
        return self.classifier(torch.cat([cnn_features, engineered], dim=1)).squeeze(-1)


def make_engineered_features(features, train_indices):
    fast_end = WLS_FAST_COUNT
    slow_end = fast_end + WLS_SLOW_COUNT
    edge_end = slow_end + ED_COUNT
    cal_end = edge_end + CAL_COUNT

    fast = features[:, :fast_end]
    slow = features[:, fast_end:slow_end]
    edge = features[:, slow_end:edge_end]
    cal = features[:, edge_end:cal_end]

    total = features.sum(dim=1)
    fast_total = fast.sum(dim=1)
    slow_total = slow.sum(dim=1)
    edge_total = edge.sum(dim=1)
    cal_total = cal.sum(dim=1)
    wls_total = fast_total + slow_total
    active_wls = ((fast > 0) | (slow > 0)).sum(dim=1).float()

    engineered = torch.stack(
        [
            total,
            fast_total,
            slow_total,
            edge_total,
            cal_total,
            active_wls,
            slow_total / torch.clamp(wls_total, min=1e-9),
            edge_total / torch.clamp(total, min=1e-9),
            cal_total / torch.clamp(total, min=1e-9),
        ],
        dim=1,
    )

    mean = engineered[train_indices].mean(dim=0)
    std = engineered[train_indices].std(dim=0)
    std[std < 1e-6] = 1.0
    return (engineered - mean) / std


def train_hybrid_cnn(features, labels, train_indices, valid_indices, test_indices, pos_weight):
    wls, small = make_detector_tensors(features)
    engineered = make_engineered_features(features, train_indices)
    dataset = TensorDataset(wls, small, engineered, labels)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    model = PairEventHybridCNN(engineered.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_model("Hybrid CNN", model, train_loader, valid_loader, criterion, optimizer)
    return evaluate_model(model, test_loader)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_hybrid_cnn.py <datafile1> [<datafile2> ...]")
        return

    torch.manual_seed(SEED)
    features, labels = load_events(sys.argv[1:])
    train_indices, valid_indices, test_indices = split_indices(len(labels))
    pos_weight = train_pos_weight(labels, train_indices)

    print(f"Loaded {len(labels)} events")
    print(f"Split: {len(train_indices)} train, {len(valid_indices)} validation, {len(test_indices)} test")
    print(f"Train pos_weight: {pos_weight.item():.3f}\n")

    mlp_metrics = train_mlp_baseline(features, labels, train_indices, valid_indices, test_indices, pos_weight)
    hybrid_metrics = train_hybrid_cnn(features, labels, train_indices, valid_indices, test_indices, pos_weight)

    print_metrics("Original MLP", mlp_metrics)
    print_metrics("Hybrid CNN", hybrid_metrics)
    print(f"\nAccuracy difference: {hybrid_metrics[0] - mlp_metrics[0]:+.3f} percentage points")


if __name__ == "__main__":
    main()
