import argparse
import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib-cache-pair-detection"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "font-cache-pair-detection"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from EventDataset import (
    CAL_COUNT,
    ED_COUNT,
    EventDataset,
    WLS_FAST_COUNT,
    WLS_SLOW_COUNT,
)


PAIR_COLOR = "tab:orange"
NON_PAIR_COLOR = "tab:blue"
DIFF_CMAP = "coolwarm"
SIGNAL_CMAP = "viridis"
EPSILON = 1e-9


def load_events(paths):
    features = []
    labels = []

    for path in paths:
        dataset = EventDataset(path)
        features.append(dataset.features_tensor.numpy())
        labels.append(dataset.labels_tensor.numpy().astype(int))

    return np.vstack(features), np.concatenate(labels)


def split_features(features):
    fast_start = 0
    slow_start = fast_start + WLS_FAST_COUNT
    edge_start = slow_start + WLS_SLOW_COUNT
    cal_start = edge_start + ED_COUNT

    return {
        "fast": features[:, fast_start:slow_start],
        "slow": features[:, slow_start:edge_start],
        "edge": features[:, edge_start:cal_start],
        "cal": features[:, cal_start:cal_start + CAL_COUNT],
    }


def wls_direction_map(event_features, block_name):
    parts = split_features(event_features.reshape(1, -1))
    return parts[block_name].reshape(4, 2, 75)


def compact_detector_map(event_features, block_name):
    parts = split_features(event_features.reshape(1, -1))
    return parts[block_name].reshape(4, 2, 3).reshape(4, 6)


def engineered_statistics(features):
    parts = split_features(features)
    fast = parts["fast"]
    slow = parts["slow"]
    edge = parts["edge"]
    cal = parts["cal"]

    wls_total = fast.sum(axis=1) + slow.sum(axis=1)
    total = features.sum(axis=1)

    return {
        "Total signal": total,
        "WLS fast signal": fast.sum(axis=1),
        "WLS slow signal": slow.sum(axis=1),
        "Edge detector signal": edge.sum(axis=1),
        "Calorimeter signal": cal.sum(axis=1),
        "Active WLS channels": ((fast > 0) | (slow > 0)).sum(axis=1),
        "Slow fraction of WLS signal": slow.sum(axis=1) / np.maximum(wls_total, EPSILON),
        "Calorimeter fraction of total": cal.sum(axis=1) / np.maximum(total, EPSILON),
        "Edge fraction of total": edge.sum(axis=1) / np.maximum(total, EPSILON),
    }


def class_masks(labels):
    return labels == 1, labels == 0


def save_or_show(fig, output_path, show):
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_feature_histograms(statistics, labels, outdir, show):
    pair_mask, nonpair_mask = class_masks(labels)
    keys = list(statistics.keys())

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle("Pair vs Non-Pair Feature Distributions", fontsize=16)

    for ax, key in zip(axes, keys):
        values = statistics[key]
        nonpair_values = values[nonpair_mask]
        pair_values = values[pair_mask]

        if nonpair_values.size:
            ax.hist(
                nonpair_values,
                bins=45,
                density=True,
                alpha=0.58,
                label=f"Non-pair / Compton (n={nonpair_values.size})",
                color=NON_PAIR_COLOR,
            )
        if pair_values.size:
            ax.hist(
                pair_values,
                bins=45,
                density=True,
                alpha=0.58,
                label=f"Pair (n={pair_values.size})",
                color=PAIR_COLOR,
            )

        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Normalized event density")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save_or_show(fig, os.path.join(outdir, "01_feature_histograms.png"), show)


def plot_feature_scatter(statistics, labels, outdir, show):
    pair_mask, nonpair_mask = class_masks(labels)
    plots = [
        ("Total signal", "Active WLS channels"),
        ("Edge detector signal", "Calorimeter signal"),
        ("WLS fast signal", "WLS slow signal"),
        ("Total signal", "Calorimeter fraction of total"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Pair vs Non-Pair Separation in Engineered Features", fontsize=16)

    for ax, (x_key, y_key) in zip(axes, plots):
        x = statistics[x_key]
        y = statistics[y_key]

        if nonpair_mask.any():
            ax.scatter(
                x[nonpair_mask],
                y[nonpair_mask],
                s=12,
                alpha=0.38,
                color=NON_PAIR_COLOR,
                label="Non-pair / Compton",
            )
        if pair_mask.any():
            ax.scatter(
                x[pair_mask],
                y[pair_mask],
                s=12,
                alpha=0.38,
                color=PAIR_COLOR,
                label="Pair",
            )

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend()

    fig.tight_layout()
    save_or_show(fig, os.path.join(outdir, "02_feature_scatter.png"), show)


def plot_mean_difference_bars(statistics, labels, outdir, show):
    pair_mask, nonpair_mask = class_masks(labels)
    names = []
    effects = []

    for name, values in statistics.items():
        pair_values = values[pair_mask]
        nonpair_values = values[nonpair_mask]
        if pair_values.size == 0 or nonpair_values.size == 0:
            continue

        pooled_std = np.sqrt((pair_values.var() + nonpair_values.var()) / 2)
        effect = (pair_values.mean() - nonpair_values.mean()) / max(pooled_std, EPSILON)
        names.append(name)
        effects.append(effect)

    order = np.argsort(np.abs(effects))
    names = np.array(names)[order]
    effects = np.array(effects)[order]
    colors = [PAIR_COLOR if value > 0 else NON_PAIR_COLOR for value in effects]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(names, effects, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Which Summary Features Differ Most Between Classes?")
    ax.set_xlabel("Standardized mean difference: positive means larger for pair events")
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)

    fig.tight_layout()
    save_or_show(fig, os.path.join(outdir, "03_feature_mean_differences.png"), show)


def plot_average_detector_maps(features, labels, outdir, show):
    pair_mask, nonpair_mask = class_masks(labels)
    if not pair_mask.any() or not nonpair_mask.any():
        print("Skipped average detector maps: need both pair and non-pair events.")
        return

    map_specs = [
        ("WLS fast X direction", "fast", 0),
        ("WLS fast Y direction", "fast", 1),
        ("WLS slow X direction", "slow", 0),
        ("WLS slow Y direction", "slow", 1),
    ]

    fig, axes = plt.subplots(len(map_specs), 3, figsize=(16, 11), constrained_layout=True)
    fig.suptitle("Average Detector Response: Non-Pair vs Pair", fontsize=16)

    for row, (title, block_name, direction) in enumerate(map_specs):
        nonpair_map = split_features(features[nonpair_mask])[block_name].mean(axis=0).reshape(4, 2, 75)[:, direction, :]
        pair_map = split_features(features[pair_mask])[block_name].mean(axis=0).reshape(4, 2, 75)[:, direction, :]
        diff_map = pair_map - nonpair_map
        signal_vmax = np.percentile(np.concatenate([nonpair_map.ravel(), pair_map.ravel()]), 99)
        diff_vmax = max(np.abs(diff_map).max(), EPSILON)

        panels = [
            (nonpair_map, f"Non-pair mean\n{title}", SIGNAL_CMAP, 0, signal_vmax),
            (pair_map, f"Pair mean\n{title}", SIGNAL_CMAP, 0, signal_vmax),
            (diff_map, f"Pair minus non-pair\n{title}", DIFF_CMAP, -diff_vmax, diff_vmax),
        ]

        for col, (data, panel_title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(panel_title)
            ax.set_xlabel("Strip/channel")
            ax.set_ylabel("Layer")
            ax.set_yticks(range(4))
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    save_or_show(fig, os.path.join(outdir, "04_average_detector_maps.png"), show)


def high_wls_activity_index(features, labels, class_label):
    class_indices = np.flatnonzero(labels == class_label)
    if class_indices.size == 0:
        return None

    parts = split_features(features[class_indices])
    fast = parts["fast"]
    slow = parts["slow"]
    wls_total = fast.sum(axis=1) + slow.sum(axis=1)
    active_wls = ((fast > 0) | (slow > 0)).sum(axis=1)

    summary = np.column_stack(
        [
            wls_total,
            active_wls,
        ]
    )
    summary = (summary - summary.mean(axis=0)) / np.maximum(summary.std(axis=0), EPSILON)
    activity_score = summary.sum(axis=1)
    target_score = np.percentile(activity_score, 75)
    return class_indices[np.argmin(np.abs(activity_score - target_score))]


def plot_representative_events(features, labels, outdir, show):
    nonpair_index = high_wls_activity_index(features, labels, 0)
    pair_index = high_wls_activity_index(features, labels, 1)
    if nonpair_index is None or pair_index is None:
        print("Skipped representative event display: need both pair and non-pair events.")
        return

    columns = [
        (nonpair_index, "High-WLS-activity non-pair / Compton event"),
        (pair_index, "High-WLS-activity pair event"),
    ]
    rows = [
        ("WLS fast X", lambda event: wls_direction_map(event, "fast")[:, 0, :]),
        ("WLS fast Y", lambda event: wls_direction_map(event, "fast")[:, 1, :]),
        ("WLS slow X", lambda event: wls_direction_map(event, "slow")[:, 0, :]),
        ("WLS slow Y", lambda event: wls_direction_map(event, "slow")[:, 1, :]),
        ("Edge detector\nX cells | Y cells", lambda event: compact_detector_map(event, "edge")),
        ("Calorimeter\nX cells | Y cells", lambda event: compact_detector_map(event, "cal")),
    ]

    fig, axes = plt.subplots(len(rows), len(columns), figsize=(13, 16), constrained_layout=True)
    fig.suptitle("Individual High-WLS-Activity Event Displays", fontsize=16)

    for col, (event_index, column_title) in enumerate(columns):
        event = features[event_index]
        event_total = event.sum()
        for row, (row_title, map_func) in enumerate(rows):
            data = map_func(event)
            ax = axes[row, col]
            image = ax.imshow(data, aspect="auto", cmap=SIGNAL_CMAP, vmin=0, vmax=max(data.max(), 1))
            if row == 0:
                ax.set_title(f"{column_title}\nindex={event_index}, total signal={event_total:.1f}")
            ax.set_ylabel(f"{row_title}\nLayer")
            ax.set_xlabel("Strip/channel")
            ax.set_yticks(range(4))
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    save_or_show(fig, os.path.join(outdir, "05_representative_events.png"), show)


def write_summary(statistics, labels, outdir):
    pair_mask, nonpair_mask = class_masks(labels)
    summary_path = os.path.join(outdir, "summary.txt")

    with open(summary_path, "w") as file:
        file.write("Pair vs non-pair visualization summary\n")
        file.write(f"Pair events: {pair_mask.sum()}\n")
        file.write(f"Non-pair / Compton events: {nonpair_mask.sum()}\n\n")

        for name, values in statistics.items():
            file.write(f"{name}\n")
            if nonpair_mask.any():
                file.write(f"  non-pair mean: {values[nonpair_mask].mean():.4f}\n")
            if pair_mask.any():
                file.write(f"  pair mean: {values[pair_mask].mean():.4f}\n")
            file.write("\n")

    print(f"Saved {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create labeled visual comparisons of pair and non-pair events."
    )
    parser.add_argument("datafiles", nargs="+", help="One or more event data text files.")
    parser.add_argument(
        "--outdir",
        default="pair_nonpair_plots",
        help="Directory where PNG plots will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open each plot interactively after saving it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    features, labels = load_events(args.datafiles)
    statistics = engineered_statistics(features)

    pair_count = int((labels == 1).sum())
    nonpair_count = int((labels == 0).sum())
    print(f"Loaded {len(labels)} events: {pair_count} pair, {nonpair_count} non-pair / Compton")

    plot_feature_histograms(statistics, labels, args.outdir, args.show)
    plot_feature_scatter(statistics, labels, args.outdir, args.show)
    plot_mean_difference_bars(statistics, labels, args.outdir, args.show)
    plot_average_detector_maps(features, labels, args.outdir, args.show)
    plot_representative_events(features, labels, args.outdir, args.show)
    write_summary(statistics, labels, args.outdir)


if __name__ == "__main__":
    main()
