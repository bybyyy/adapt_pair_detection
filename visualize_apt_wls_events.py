"""Visualize held-out APT WLS maps and summarize CNN failure patterns by layer."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from EventDataset import AptPairDataset
from train_2d_cnn import PairEventAPT2DCNN
from train_apt_pair_models import campaign_split


CHANNEL_NAMES = ["WLS fast X", "WLS fast Y", "WLS slow X", "WLS slow Y"]
GROUP_LABELS = {
    "true_positive": "Pair, correctly predicted",
    "false_negative": "Pair, predicted non-pair",
    "true_negative": "Non-pair, correctly predicted",
    "false_positive": "Non-pair, predicted pair",
}


def import_plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apt-pair-matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [
            Path("/usr/local/lib") / version / "dist-packages",
            Path(sys.base_prefix) / "local/lib" / version / "dist-packages",
            Path(sys.base_prefix) / "lib" / version / "dist-packages",
        ]
        for system_packages in candidates:
            if system_packages.is_dir():
                sys.path.append(str(system_packages))
        import matplotlib.pyplot as plt
    return plt


def predict_test_set(dataset, checkpoint_path, batch_size):
    _, _, test_indices = campaign_split(
        dataset.labels,
        dataset.energy_mev,
        dataset.run_ids,
        dataset.random_seeds,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PairEventAPT2DCNN()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = float(checkpoint["threshold"])
    loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits = []
    with torch.no_grad():
        for wls, _, _ in loader:
            logits.append(model(wls).cpu().numpy())
    logits = np.concatenate(logits)
    indices = test_indices.numpy()
    labels = dataset.labels[indices].astype(np.int8)
    predicted = (logits > threshold).astype(np.int8)
    return indices, labels, predicted, logits, threshold


def classification_group(actual, predicted):
    if actual == 1 and predicted == 1:
        return "true_positive"
    if actual == 1:
        return "false_negative"
    if predicted == 0:
        return "true_negative"
    return "false_positive"


def sample_group(indices, dataset, actual, predicted, group, n10, n15, rng):
    mask = np.asarray(
        [classification_group(a, p) == group for a, p in zip(actual, predicted)]
    )
    candidates = indices[mask]
    chosen = []
    for energy, count in [(10.0, n10), (15.0, n15)]:
        energy_candidates = candidates[dataset.energy_mev[candidates] == energy]
        if len(energy_candidates) < count:
            raise ValueError(
                f"Need {count} {group} examples at {energy:g} MeV; "
                f"found {len(energy_candidates)}"
            )
        chosen.extend(rng.choice(energy_candidates, size=count, replace=False).tolist())
    return chosen


def build_selection(indices, labels, predicted, dataset, seed):
    rng = np.random.default_rng(seed)
    by_group = {
        "true_positive": sample_group(
            indices, dataset, labels, predicted, "true_positive", 3, 2, rng
        ),
        "false_negative": sample_group(
            indices, dataset, labels, predicted, "false_negative", 2, 3, rng
        ),
        "true_negative": sample_group(
            indices, dataset, labels, predicted, "true_negative", 3, 2, rng
        ),
        "false_positive": sample_group(
            indices, dataset, labels, predicted, "false_positive", 2, 3, rng
        ),
    }
    return by_group


def event_record(dataset, index, predicted, logit, threshold, group):
    run_id, event_id = dataset.sample_key(index)
    return {
        "dataset_index": int(index),
        "run_id": run_id,
        "event_id": event_id,
        "energy_mev": float(dataset.energy_mev[index]),
        "actual_label": int(dataset.labels[index]),
        "predicted_label": int(predicted),
        "logit": float(logit),
        "threshold": threshold,
        "group": group,
    }


def plot_event(plt, tensor, record, output_path, vmax_by_channel):
    transformed = np.log1p(np.maximum(tensor, 0.0))
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for channel, axis in enumerate(axes.flat):
        image = axis.imshow(
            transformed[channel],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, tensor.shape[2] - 0.5, -0.5, tensor.shape[1] - 0.5),
            cmap="magma",
            vmin=0.0,
            vmax=vmax_by_channel[channel],
        )
        axis.set_title(CHANNEL_NAMES[channel])
        axis.set_xlabel("Fiber channel")
        axis.set_ylabel("Detector layer")
        axis.set_yticks([0, 5, 10, 15, 19])
        figure.colorbar(image, ax=axis, label="log(1 + photoelectrons)", shrink=0.86)
    actual = "pair" if record["actual_label"] else "non-pair"
    prediction = "pair" if record["predicted_label"] else "non-pair"
    figure.suptitle(
        f"APT WLS event: actual {actual}, predicted {prediction} | "
        f"{record['energy_mev']:g} MeV | {record['run_id']} event {record['event_id']}\n"
        "Layer 19 is source-facing; layer 0 is farthest from the source",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_contact_sheet(plt, dataset, records, output_path, vmax_by_channel, title):
    if len(records) != 10:
        raise ValueError(f"Contact sheets require ten records, got {len(records)}")
    correct = records[:5]
    incorrect = records[5:]
    figure, axes = plt.subplots(5, 8, figsize=(24, 13))
    for row, paired_records in enumerate(zip(correct, incorrect)):
        for side, record in enumerate(paired_records):
            tensor = dataset[record["dataset_index"]][0].numpy()
            transformed = np.log1p(np.maximum(tensor, 0.0))
            for channel in range(4):
                column = side * 4 + channel
                axis = axes[row, column]
                axis.imshow(
                    transformed[channel],
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax_by_channel[channel],
                )
                if row == 0:
                    status = "Correct" if side == 0 else "Misclassified"
                    axis.set_title(f"{status}\n{CHANNEL_NAMES[channel]}", fontsize=10)
                if channel == 0:
                    axis.set_ylabel(
                        f"{record['energy_mev']:g} MeV\nevent {record['event_id']}",
                        fontsize=8,
                    )
                    axis.set_yticks([0, 10, 19])
                else:
                    axis.set_yticks([])
                axis.set_xticks([0, 746, 1491] if row == 4 else [])
    figure.suptitle(
        f"{title}\nLayer 19 is source-facing; layer 0 is farthest from the source",
        fontsize=15,
    )
    figure.supxlabel("Fiber channel")
    figure.subplots_adjust(
        left=0.045, right=0.995, bottom=0.06, top=0.91, wspace=0.08, hspace=0.16
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def event_layer_metrics(tensor):
    layer_signal = np.asarray(tensor, dtype=np.float64).sum(axis=(0, 2))
    total = float(layer_signal.sum())
    if total <= 0:
        return layer_signal, {
            "total_signal": 0.0,
            "active_layers": 0,
            "layer_centroid": float("nan"),
            "far_five_fraction": float("nan"),
            "source_five_fraction": float("nan"),
        }
    return layer_signal / total, {
        "total_signal": total,
        "active_layers": int(np.count_nonzero(layer_signal)),
        "layer_centroid": float(np.dot(np.arange(20), layer_signal) / total),
        "far_five_fraction": float(layer_signal[:5].sum() / total),
        "source_five_fraction": float(layer_signal[15:].sum() / total),
    }


def summarize_groups(dataset, indices, labels, predicted):
    profiles = {key: [] for key in GROUP_LABELS}
    metrics = {key: [] for key in GROUP_LABELS}
    for index, actual, prediction in zip(indices, labels, predicted):
        group = classification_group(actual, prediction)
        tensor = dataset[int(index)][0].numpy()
        profile, event_metrics = event_layer_metrics(tensor)
        profiles[group].append(profile)
        metrics[group].append(event_metrics)
    profiles = {key: np.asarray(value) for key, value in profiles.items()}
    summary = {}
    for group, rows in metrics.items():
        summary[group] = {"events": len(rows)}
        for metric in rows[0]:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[group][metric] = {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
            }
    return profiles, summary


def bootstrap_profile(profile, rng, samples=1000):
    means = np.empty((samples, profile.shape[1]), dtype=float)
    for sample in range(samples):
        chosen = rng.integers(0, len(profile), size=len(profile))
        means[sample] = profile[chosen].mean(axis=0)
    return profile.mean(axis=0), np.percentile(means, [2.5, 97.5], axis=0)


def plot_layer_profiles(plt, profiles, output_path, seed):
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, constrained_layout=True)
    comparisons = [
        ("Pair events", "true_positive", "false_negative"),
        ("Non-pair events", "true_negative", "false_positive"),
    ]
    styles = [("#356AA0", "-"), ("#D17A22", "--")]
    layers = np.arange(20)
    for axis, (title, first, second) in zip(axes, comparisons):
        for group, (color, linestyle) in zip([first, second], styles):
            mean, interval = bootstrap_profile(profiles[group], rng)
            axis.plot(
                mean,
                layers,
                label=f"{GROUP_LABELS[group]} (n={len(profiles[group])})",
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            axis.fill_betweenx(layers, interval[0], interval[1], color=color, alpha=0.18)
        axis.set_title(title)
        axis.set_xlabel("Mean fraction of event WLS signal")
        axis.grid(True, color="#d8d8d8", linewidth=0.6)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Detector layer (19 source-facing; 0 farthest)")
    axes[0].set_yticks(range(20))
    figure.suptitle("APT test-set WLS activity by layer (mean and bootstrap 95% CI)")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_readme(path, records, summary, threshold):
    pair_correct = sum(item["group"] == "true_positive" for item in records)
    pair_failed = sum(item["group"] == "false_negative" for item in records)
    nonpair_correct = sum(item["group"] == "true_negative" for item in records)
    nonpair_failed = sum(item["group"] == "false_positive" for item in records)
    lines = [
        "# APT WLS event images",
        "",
        "These images use the held-out seed-ending-9 test runs and the validation-selected CNN threshold.",
        "Each event contains four maps: WLS fast/slow signals in the X/Y directions. The horizontal axis",
        "is fiber channel and the vertical axis is detector layer. Layer 19 is source-facing and layer 0",
        "is farthest from the source for this fixed -z source configuration.",
        "",
        "Signal color is `log(1 + photoelectrons)`. Each WLS channel uses a fixed scale across all 20",
        "selected events, so examples are visually comparable without bright outliers setting the scale.",
        "",
        f"The deterministic sample contains {pair_correct} correctly classified pair events, {pair_failed}",
        f"false negatives, {nonpair_correct} correctly classified non-pair events, and {nonpair_failed}",
        "false positives, stratified across 10 and 15 MeV.",
        "",
        f"CNN logit threshold: `{threshold:.8f}`.",
        "",
        "- `pair_contact_sheet.png`: ten true pair events (five correct, five false negatives).",
        "- `nonpair_contact_sheet.png`: ten true non-pair events (five correct, five false positives).",
        "- `layer_failure_profiles.png`: all test events, comparing normalized layer activity for correct",
        "  and incorrect predictions with bootstrap 95% confidence intervals.",
        "- `selected_events.json`: exact run/event IDs and predictions for the displayed examples.",
        "- `failure_summary.json`: layer-location and signal summaries over the full test set.",
        "",
        "## Full-test-set layer centroid medians",
        "",
    ]
    for group in ["true_positive", "false_negative", "true_negative", "false_positive"]:
        centroid = summary[group]["layer_centroid"]["median"]
        lines.append(f"- {GROUP_LABELS[group]}: `{centroid:.2f}` (n={summary[group]['events']})")
    path.write_text("\n".join(lines) + "\n")


def main(args):
    plt = import_plotting()
    dataset = AptPairDataset(args.manifest)
    indices, labels, predicted, logits, threshold = predict_test_set(
        dataset, args.checkpoint, args.batch_size
    )
    output_dir = args.output_dir.resolve()
    pair_dir = output_dir / "pair"
    nonpair_dir = output_dir / "nonpair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    nonpair_dir.mkdir(parents=True, exist_ok=True)

    selection = build_selection(indices, labels, predicted, dataset, args.seed)
    position = {int(index): row for row, index in enumerate(indices)}
    records = []
    ordered_groups = ["true_positive", "false_negative", "true_negative", "false_positive"]
    for group in ordered_groups:
        for index in selection[group]:
            row = position[int(index)]
            records.append(
                event_record(
                    dataset,
                    int(index),
                    int(predicted[row]),
                    float(logits[row]),
                    threshold,
                    group,
                )
            )

    selected_tensors = {
        item["dataset_index"]: dataset[item["dataset_index"]][0].numpy() for item in records
    }
    vmax_by_channel = []
    for channel in range(4):
        positive = np.concatenate(
            [
                np.log1p(np.maximum(tensor[channel], 0.0))[tensor[channel] > 0]
                for tensor in selected_tensors.values()
                if np.any(tensor[channel] > 0)
            ]
        )
        vmax_by_channel.append(float(np.percentile(positive, 99.5)))

    class_counters = {"pair": 0, "nonpair": 0}
    for record in records:
        truth = "pair" if record["actual_label"] else "nonpair"
        class_counters[truth] += 1
        status = (
            "correct"
            if record["actual_label"] == record["predicted_label"]
            else "misclassified"
        )
        filename = (
            f"{truth}_{class_counters[truth]:02d}_{status}_"
            f"{record['energy_mev']:g}MeV_{record['run_id']}_event_{record['event_id']}.png"
        )
        plot_event(
            plt,
            selected_tensors[record["dataset_index"]],
            record,
            output_dir / truth / filename,
            vmax_by_channel,
        )

    pair_records = [item for item in records if item["actual_label"] == 1]
    nonpair_records = [item for item in records if item["actual_label"] == 0]
    plot_contact_sheet(
        plt,
        dataset,
        pair_records,
        output_dir / "pair_contact_sheet.png",
        vmax_by_channel,
        "True pair events: five correct and five false negatives",
    )
    plot_contact_sheet(
        plt,
        dataset,
        nonpair_records,
        output_dir / "nonpair_contact_sheet.png",
        vmax_by_channel,
        "True non-pair events: five correct and five false positives",
    )

    profiles, summary = summarize_groups(dataset, indices, labels, predicted)
    plot_layer_profiles(
        plt, profiles, output_dir / "layer_failure_profiles.png", args.seed + 1
    )
    (output_dir / "selected_events.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "threshold": threshold,
                "selection_seed": args.seed,
                "vmax_log1p_by_channel": dict(zip(CHANNEL_NAMES, vmax_by_channel)),
                "events": records,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "failure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_readme(output_dir / "README.md", records, summary, threshold)
    print(f"Wrote {len(pair_records)} pair and {len(nonpair_records)} non-pair event images")
    print(f"Wrote contact sheets and layer summary to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    campaign = Path("/ssd_data/boran.y/apt_pair_detection_exploratory_v2")
    parser.add_argument("--manifest", type=Path, default=campaign / "datasets/manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=campaign / "results/apt_cnn_main.pt")
    parser.add_argument(
        "--output-dir", type=Path, default=campaign / "results/wls_event_images"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
