"""Train and evaluate WLS-only APT pair classifiers from an APT manifest."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from EventDataset import AptPairDataset
from train_2d_cnn import PairEventAPT2DCNN
from train_hybrid_cnn import PairEventAPTHybridCNN


SEED = 42


def allocate_run_groups(run_ids, rng, partitions=3):
    groups = np.asarray(sorted(set(run_ids.tolist())), dtype=object)
    if len(groups) < partitions:
        raise ValueError(
            f"Need at least {partitions} independent seed-runs per energy; found {len(groups)}"
        )
    rng.shuffle(groups)
    if partitions == 2:
        valid_count = max(1, int(round(0.1 * len(groups))))
        return groups[valid_count:], groups[:valid_count]
    valid_count = max(1, int(round(0.1 * len(groups))))
    test_count = max(1, int(round(0.1 * len(groups))))
    if valid_count + test_count >= len(groups):
        valid_count = test_count = 1
    return groups[valid_count + test_count :], groups[:valid_count], groups[valid_count : valid_count + test_count]


def validate_partition_classes(partitions, labels, energies):
    for partition_name, indices in zip(["train", "validation", "test"], partitions):
        selected_labels = labels[indices]
        for energy in np.unique(energies[indices]):
            class_count = len(np.unique(selected_labels[energies[indices] == energy]))
            if class_count != 2:
                raise ValueError(
                    f"{partition_name} partition lacks both labels at {energy:g} MeV; "
                    "generate additional independent seed-runs"
                )


def stratified_split(labels, energies, run_ids, seed=SEED):
    """Split whole runs within every energy; no run can cross a partition."""
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for energy in sorted(np.unique(energies).tolist()):
        energy_mask = energies == energy
        allocated_runs = allocate_run_groups(run_ids[energy_mask], rng)
        for destination, groups in zip(partitions, allocated_runs):
            values = np.flatnonzero(energy_mask & np.isin(run_ids, groups))
            destination.extend(values.tolist())
    for values in partitions:
        rng.shuffle(values)
    arrays = tuple(np.asarray(values, dtype=np.int64) for values in partitions)
    validate_partition_classes(arrays, labels, energies)
    return tuple(torch.tensor(values, dtype=torch.long) for values in arrays)


def leave_one_energy_out_split(labels, energies, run_ids, held_energy, seed=SEED):
    rng = np.random.default_rng(seed + int(round(float(held_energy) * 10)))
    test = np.flatnonzero(energies == held_energy)
    train = []
    valid = []
    for energy in sorted(np.unique(energies[energies != held_energy]).tolist()):
        energy_mask = energies == energy
        train_runs, valid_runs = allocate_run_groups(run_ids[energy_mask], rng, partitions=2)
        train.extend(np.flatnonzero(energy_mask & np.isin(run_ids, train_runs)).tolist())
        valid.extend(np.flatnonzero(energy_mask & np.isin(run_ids, valid_runs)).tolist())
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    arrays = [np.asarray(values, dtype=np.int64) for values in [train, valid, test]]
    for name, values in zip(["train", "validation", "test"], arrays):
        if len(np.unique(labels[values])) != 2:
            raise ValueError(f"{name} leave-one-energy-out partition lacks both labels")
    return tuple(torch.tensor(values, dtype=torch.long) for values in arrays)


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def class_weight(labels, train_indices, device):
    selected = labels[train_indices.numpy()]
    positives = int(selected.sum())
    negatives = len(selected) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training partition must contain both classes")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_model(model, loaders, labels, train_indices, device, epochs, lr, weight_decay):
    train_loader, valid_loader, _ = loaders
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight(labels, train_indices, device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for wls, batch_labels, _ in train_loader:
            wls = wls.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(wls), batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for wls, batch_labels, _ in valid_loader:
                wls = wls.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                valid_loss += criterion(model(wls), batch_labels).item()
        train_loss /= max(len(train_loader), 1)
        valid_loss /= max(len(valid_loader), 1)
        print(
            f"epoch {epoch + 1:02d}: train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_loss


def collect_predictions(model, loader, device):
    logits = []
    labels = []
    energies = []
    model.eval()
    with torch.no_grad():
        for wls, batch_labels, batch_energies in loader:
            logits.append(model(wls.to(device, non_blocking=True)).cpu())
            labels.append(batch_labels.cpu())
            energies.append(batch_energies.cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(energies)


def confusion(logits, labels, threshold):
    predicted = (logits > threshold).int()
    labels = labels.int()
    return (
        int(((predicted == 1) & (labels == 1)).sum()),
        int(((predicted == 1) & (labels == 0)).sum()),
        int(((predicted == 0) & (labels == 1)).sum()),
        int(((predicted == 0) & (labels == 0)).sum()),
    )


def metrics_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": 100.0 * (tp + tn) / total if total else 0.0,
        "balanced_accuracy": 50.0 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def select_threshold(logits, labels):
    if torch.equal(logits.min(), logits.max()):
        return float(logits.min())
    best = (-1.0, 0.0)
    for threshold in torch.linspace(logits.min(), logits.max(), 401).tolist():
        score = metrics_from_counts(*confusion(logits, labels, threshold))["balanced_accuracy"]
        if score > best[0]:
            best = (score, threshold)
    return float(best[1])


def evaluate(model, valid_loader, test_loader, device):
    valid_logits, valid_labels, _ = collect_predictions(model, valid_loader, device)
    threshold = select_threshold(valid_logits, valid_labels)
    logits, labels, energies = collect_predictions(model, test_loader, device)
    result = {
        "threshold": threshold,
        "overall": metrics_from_counts(*confusion(logits, labels, threshold)),
        "per_energy": {},
    }
    for energy in sorted(torch.unique(energies).tolist()):
        mask = energies == energy
        result["per_energy"][f"{energy:g}"] = metrics_from_counts(
            *confusion(logits[mask], labels[mask], threshold)
        )
    return result


def create_model(name):
    if name == "cnn":
        return PairEventAPT2DCNN()
    if name == "hybrid":
        return PairEventAPTHybridCNN()
    raise ValueError(name)


def run_experiment(name, dataset, indices, args, device, tag):
    train_indices, valid_indices, test_indices = indices
    loaders = (
        make_loader(dataset, train_indices, args.batch_size, True),
        make_loader(dataset, valid_indices, args.batch_size, False),
        make_loader(dataset, test_indices, args.batch_size, False),
    )
    torch.manual_seed(args.seed)
    model = create_model(name).to(device)
    print(f"\nTraining {name} ({tag})")
    valid_loss = train_model(
        model,
        loaders,
        dataset.labels,
        train_indices,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
    )
    result = evaluate(model, loaders[1], loaders[2], device)
    result["valid_loss"] = valid_loss
    result["partition_counts"] = {
        "train": len(train_indices),
        "validation": len(valid_indices),
        "test": len(test_indices),
    }
    checkpoint = args.outdir / f"apt_{name}_{tag}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": name,
            "geometry": dataset.geometry,
            "threshold": result["threshold"],
        },
        checkpoint,
    )
    result["checkpoint"] = str(checkpoint)
    return result


def write_report(payload, path):
    lines = [
        "# APT WLS-only Pair Detection Report",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Events: `{payload['events']}`",
        f"- Pair events: `{payload['pair_events']}`",
        f"- Non-pair events: `{payload['nonpair_events']}`",
        f"- Random seed: `{payload['seed']}`",
        "",
        "## Main stratified evaluation",
        "",
    ]
    for name, result in payload["main"].items():
        metrics = result["overall"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Balanced accuracy: `{metrics['balanced_accuracy']:.3f}%`",
                f"- Accuracy: `{metrics['accuracy']:.3f}%`",
                f"- Precision: `{metrics['precision']:.4f}`",
                f"- Recall: `{metrics['recall']:.4f}`",
                f"- F1: `{metrics['f1']:.4f}`",
                f"- Confusion counts: `TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}`",
                "",
                "| Energy (MeV) | Events | Pair | Non-pair | Balanced accuracy | Precision | Recall | F1 | Confusion (TP/FP/FN/TN) |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for energy, energy_metrics in result["per_energy"].items():
            counts = payload["class_counts"][energy]
            lines.append(
                f"| {energy} | {counts['events']} | {counts['pair']} | {counts['nonpair']} | "
                f"{energy_metrics['balanced_accuracy']:.3f}% | {energy_metrics['precision']:.4f} | "
                f"{energy_metrics['recall']:.4f} | {energy_metrics['f1']:.4f} | "
                f"{energy_metrics['tp']}/{energy_metrics['fp']}/{energy_metrics['fn']}/{energy_metrics['tn']} |"
            )
        lines.append("")
    if payload["leave_one_energy_out"]:
        lines.extend(["## Leave-one-energy-out evaluation", ""])
        for tag, models in payload["leave_one_energy_out"].items():
            lines.append(f"### {tag}")
            lines.append("")
            for name, result in models.items():
                metrics = result["overall"]
                lines.append(
                    f"- {name}: balanced accuracy `{metrics['balanced_accuracy']:.3f}%`, "
                    f"F1 `{metrics['f1']:.4f}`"
                )
            lines.append("")
    lines.extend(
        [
            "## ADAPT reference caveat",
            "",
            (
                "The historical ADAPT benchmark reported a best balanced accuracy of 73.830% "
                "and F1 of 0.770, but it included WLS, edge-detector, and calorimeter inputs. "
                "APT results here are WLS-only and are not an apples-to-apples detector comparison."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark WLS-only APT pair classifiers.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("apt_benchmarks"))
    parser.add_argument("--models", choices=["cnn", "hybrid"], nargs="+", default=["cnn", "hybrid"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-class-count-per-energy", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-loeo", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset = AptPairDataset(args.manifest)
    if dataset.geometry["feature_shape"] != [4, 20, 1492]:
        raise ValueError(f"Expected APT feature shape [4, 20, 1492], got {dataset.geometry['feature_shape']}")

    indices = stratified_split(dataset.labels, dataset.energy_mev, dataset.run_ids, args.seed)
    class_counts = {}
    for energy in sorted(np.unique(dataset.energy_mev).tolist()):
        selected = dataset.labels[dataset.energy_mev == energy]
        pair = int(selected.sum())
        class_counts[f"{energy:g}"] = {
            "events": len(selected),
            "pair": pair,
            "nonpair": int(len(selected) - pair),
        }
        if min(pair, len(selected) - pair) < args.min_class_count_per_energy:
            raise ValueError(
                f"{energy:g} MeV has pair={pair}, non-pair={len(selected) - pair}; "
                f"both must be at least {args.min_class_count_per_energy}. "
                "Generate and append another 6,400-event seed-run."
            )
    payload = {
        "manifest": str(args.manifest.resolve()),
        "events": len(dataset),
        "pair_events": int(dataset.labels.sum()),
        "nonpair_events": int(len(dataset) - dataset.labels.sum()),
        "class_counts": class_counts,
        "seed": args.seed,
        "geometry": dataset.geometry,
        "main": {},
        "leave_one_energy_out": {},
    }
    for name in args.models:
        payload["main"][name] = run_experiment(name, dataset, indices, args, device, "main")

    if not args.skip_loeo:
        for energy in sorted(np.unique(dataset.energy_mev).tolist()):
            tag = f"holdout_{energy:g}MeV"
            held_indices = leave_one_energy_out_split(
                dataset.labels, dataset.energy_mev, dataset.run_ids, energy, args.seed
            )
            payload["leave_one_energy_out"][tag] = {}
            for name in args.models:
                payload["leave_one_energy_out"][tag][name] = run_experiment(
                    name, dataset, held_indices, args, device, tag
                )

    json_path = args.outdir / "apt_pair_detection_benchmark.json"
    report_path = args.outdir / "apt_pair_detection_report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload, report_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
