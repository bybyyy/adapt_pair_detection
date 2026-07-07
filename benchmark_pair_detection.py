import argparse
import csv
import json
import os
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from EventDataset import CAL_COUNT, ED_COUNT, WLS_FAST_COUNT, WLS_SLOW_COUNT
from PairEventClassifier import PairEventClassifier
from train_2d_cnn import (
    LR,
    SEED,
    PairEvent2DCNN,
    load_events,
    make_detector_tensors,
    split_indices,
    train_pos_weight,
)
from train_threshold_norm_tune import (
    RunConfig,
    collect_logits,
    confusion_counts,
    make_loader,
    run_experiment,
    summarize_counts,
    threshold_sweep,
    train_model,
    tune_configs,
)


DEFAULT_DATAFILES = [
    "classifier_data_5MeV.txt",
    "classifier_data_10MeV.txt",
    "classifier_data_50MeV.txt",
]

EPS = 1e-9


def engineered_statistics(features):
    fast_end = WLS_FAST_COUNT
    slow_end = fast_end + WLS_SLOW_COUNT
    edge_end = slow_end + ED_COUNT
    cal_end = edge_end + CAL_COUNT

    fast = features[:, :fast_end]
    slow = features[:, fast_end:slow_end]
    edge = features[:, slow_end:edge_end]
    cal = features[:, edge_end:cal_end]

    wls_total = fast.sum(dim=1) + slow.sum(dim=1)
    total = features.sum(dim=1)

    return {
        "Total signal": total,
        "WLS fast signal": fast.sum(dim=1),
        "WLS slow signal": slow.sum(dim=1),
        "Edge detector signal": edge.sum(dim=1),
        "Calorimeter signal": cal.sum(dim=1),
        "Active WLS channels": ((fast > 0) | (slow > 0)).sum(dim=1).float(),
        "Slow fraction of WLS signal": slow.sum(dim=1) / torch.clamp(wls_total, min=EPS),
        "Calorimeter fraction of total": cal.sum(dim=1) / torch.clamp(total, min=EPS),
        "Edge fraction of total": edge.sum(dim=1) / torch.clamp(total, min=EPS),
    }


def class_counts(labels):
    pair_count = int((labels == 1).sum().item())
    nonpair_count = int((labels == 0).sum().item())
    return pair_count, nonpair_count


def split_counts(labels, indices):
    counts = {}
    for name, split_indices_for_name in zip(["train", "validation", "test"], indices):
        split_labels = labels[split_indices_for_name]
        pair_count, nonpair_count = class_counts(split_labels)
        counts[name] = {
            "total": len(split_labels),
            "pair": pair_count,
            "nonpair": nonpair_count,
        }
    return counts


def metric_delta(newer, older, key):
    return newer["test_metrics"][key] - older["test_metrics"][key]


def result_record(
    name,
    model_type,
    threshold_metric,
    selected_threshold,
    valid_loss,
    valid_metrics,
    test_metrics,
    config=None,
    tuned=False,
):
    return {
        "name": name,
        "model_type": model_type,
        "tuned": tuned,
        "threshold_metric": threshold_metric,
        "selected_threshold": selected_threshold,
        "valid_loss": valid_loss,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "config": asdict(config) if config else {},
    }


def train_mlp_baseline(features, labels, indices, pos_weight, epochs, threshold_metric):
    train_indices, valid_indices, test_indices = indices
    dataset = TensorDataset(features, labels)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    model = PairEventClassifier(features.shape[1], pos_weight=pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    valid_loss = train_model(
        "Original MLP baseline",
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        epochs,
    )
    valid_logits, valid_labels = collect_logits(model, valid_loader)
    test_logits, test_labels = collect_logits(model, test_loader)

    threshold, _, tuned_valid_metrics = threshold_sweep(valid_logits, valid_labels, threshold_metric)
    zero_valid_metrics = summarize_counts(*confusion_counts(valid_logits, valid_labels, 0.0))
    zero_test_metrics = summarize_counts(*confusion_counts(test_logits, test_labels, 0.0))

    return result_record(
        name="Original MLP baseline",
        model_type="mlp",
        threshold_metric="fixed_zero",
        selected_threshold=0.0,
        valid_loss=valid_loss,
        valid_metrics=zero_valid_metrics,
        test_metrics=zero_test_metrics,
        tuned=False,
    ), result_record(
        name="Original MLP baseline + tuned threshold",
        model_type="mlp",
        threshold_metric=threshold_metric,
        selected_threshold=threshold,
        valid_loss=valid_loss,
        valid_metrics=tuned_valid_metrics,
        test_metrics=summarize_counts(*confusion_counts(test_logits, test_labels, threshold)),
        tuned=True,
    )


def train_original_2d_cnn(features, labels, indices, pos_weight, epochs, threshold_metric):
    train_indices, valid_indices, test_indices = indices
    wls, small = make_detector_tensors(features)
    dataset = TensorDataset(wls, small, labels)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    valid_loader = make_loader(dataset, valid_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    model = PairEvent2DCNN()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    valid_loss = train_model(
        "Original 2D CNN",
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        epochs,
    )
    valid_logits, valid_labels = collect_logits(model, valid_loader)
    test_logits, test_labels = collect_logits(model, test_loader)

    threshold, _, tuned_valid_metrics = threshold_sweep(valid_logits, valid_labels, threshold_metric)
    zero_valid_metrics = summarize_counts(*confusion_counts(valid_logits, valid_labels, 0.0))
    zero_test_metrics = summarize_counts(*confusion_counts(test_logits, test_labels, 0.0))

    return result_record(
        name="Original 2D CNN",
        model_type="2d",
        threshold_metric="fixed_zero",
        selected_threshold=0.0,
        valid_loss=valid_loss,
        valid_metrics=zero_valid_metrics,
        test_metrics=zero_test_metrics,
        tuned=False,
    ), result_record(
        name="Original 2D CNN + tuned threshold",
        model_type="2d",
        threshold_metric=threshold_metric,
        selected_threshold=threshold,
        valid_loss=valid_loss,
        valid_metrics=tuned_valid_metrics,
        test_metrics=summarize_counts(*confusion_counts(test_logits, test_labels, threshold)),
        tuned=True,
    )


def run_configured_experiment(
    model_type,
    features,
    labels,
    indices,
    pos_weight,
    config,
    threshold_metric,
    name_prefix,
    tuned,
):
    result = run_experiment(model_type, features, labels, indices, pos_weight, config, threshold_metric)
    return result_record(
        name=f"{name_prefix}: {result['name']}",
        model_type=model_type,
        threshold_metric=threshold_metric,
        selected_threshold=result["threshold"],
        valid_loss=result["valid_loss"],
        valid_metrics=result["valid_metrics"],
        test_metrics=result["tuned_test_metrics"],
        config=config,
        tuned=tuned,
    )


def select_best(results, model_type=None, tuned=None):
    candidates = results
    if model_type is not None:
        candidates = [item for item in candidates if item["model_type"] == model_type]
    if tuned is not None:
        candidates = [item for item in candidates if item["tuned"] == tuned]
    return max(
        candidates,
        key=lambda item: (
            item["test_metrics"]["balanced_accuracy"],
            item["test_metrics"]["f1"],
            item["test_metrics"]["accuracy"],
        ),
    )


def summarize_feature_statistics(features, labels):
    stats = engineered_statistics(features)
    pair_mask = labels == 1
    nonpair_mask = labels == 0
    summary = []
    for name, values in stats.items():
        pair_mean = float(values[pair_mask].mean()) if pair_mask.any() else 0.0
        nonpair_mean = float(values[nonpair_mask].mean()) if nonpair_mask.any() else 0.0
        summary.append(
            {
                "feature": name,
                "pair_mean": pair_mean,
                "nonpair_mean": nonpair_mean,
                "ratio": pair_mean / nonpair_mean if nonpair_mean else 0.0,
            }
        )
    return summary


def metric_table_row(result):
    metrics = result["test_metrics"]
    return [
        result["name"],
        result["model_type"],
        "yes" if result["tuned"] else "no",
        result["threshold_metric"],
        f"{result['selected_threshold']:.4f}",
        f"{metrics['accuracy']:.3f}",
        f"{metrics['balanced_accuracy']:.3f}",
        f"{metrics['precision']:.3f}",
        f"{metrics['recall']:.3f}",
        f"{metrics['f1']:.3f}",
        str(metrics["tp"]),
        str(metrics["fp"]),
        str(metrics["fn"]),
        str(metrics["tn"]),
    ]


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv(path, results):
    headers = [
        "name",
        "model_type",
        "tuned",
        "threshold_metric",
        "selected_threshold",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for result in results:
            writer.writerow(metric_table_row(result))


def write_json(path, payload):
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_markdown_report(path, datafiles, seed, split_summary, feature_summary, results):
    pair_total = sum(split_summary[name]["pair"] for name in split_summary)
    nonpair_total = sum(split_summary[name]["nonpair"] for name in split_summary)
    best_overall = select_best(results)
    best_hybrid = select_best(results, model_type="hybrid")
    original_2d = next(item for item in results if item["name"] == "Original 2D CNN")

    feature_rows = [
        [
            item["feature"],
            f"{item['nonpair_mean']:.4f}",
            f"{item['pair_mean']:.4f}",
            f"{item['ratio']:.3f}",
        ]
        for item in feature_summary
    ]

    result_rows = [metric_table_row(result) for result in results]
    hybrid_delta_balanced = metric_delta(best_hybrid, original_2d, "balanced_accuracy")
    hybrid_delta_f1 = metric_delta(best_hybrid, original_2d, "f1")
    errors = best_overall["test_metrics"]
    dominant_error = "FP" if errors["fp"] >= errors["fn"] else "FN"

    lines = [
        "# Pair Event Detection Stage Report",
        "",
        "## Dataset",
        "",
        f"- Data files: `{', '.join(datafiles)}`",
        f"- Random seed: `{seed}`",
        f"- Events: `{pair_total + nonpair_total}` total, `{pair_total}` pair, `{nonpair_total}` non-pair",
        "",
        markdown_table(
            ["split", "total", "pair", "nonpair"],
            [
                [
                    name,
                    str(split_summary[name]["total"]),
                    str(split_summary[name]["pair"]),
                    str(split_summary[name]["nonpair"]),
                ]
                for name in ["train", "validation", "test"]
            ],
        ),
        "",
        "## Pair vs Non-Pair Feature Differences",
        "",
        markdown_table(["feature", "nonpair mean", "pair mean", "pair/nonpair ratio"], feature_rows),
        "",
        "## Model Benchmark",
        "",
        markdown_table(
            [
                "name",
                "type",
                "tuned",
                "threshold metric",
                "threshold",
                "accuracy",
                "balanced accuracy",
                "precision",
                "recall",
                "f1",
                "TP",
                "FP",
                "FN",
                "TN",
            ],
            result_rows,
        ),
        "",
        "## Current Conclusion",
        "",
        (
            f"- Best current model by balanced accuracy/F1: `{best_overall['name']}` "
            f"with balanced accuracy `{best_overall['test_metrics']['balanced_accuracy']:.3f}%` "
            f"and F1 `{best_overall['test_metrics']['f1']:.3f}`."
        ),
        (
            f"- Its dominant test-set error type is `{dominant_error}` "
            f"(FP={errors['fp']}, FN={errors['fn']})."
        ),
        (
            f"- Best hybrid vs original 2D CNN: balanced accuracy delta "
            f"`{hybrid_delta_balanced:+.3f}` points, F1 delta `{hybrid_delta_f1:+.3f}`."
        ),
        "",
        "## Next Improvement Targets",
        "",
        "- Keep reporting balanced accuracy and F1 as primary metrics; use overall accuracy only as a secondary check.",
        "- Prefer validation-selected thresholds over threshold 0 when the physics objective tolerates threshold tuning.",
        (
            f"- Continue from the best hybrid setting observed here: "
            f"`{best_hybrid['name']}`."
        ),
        "- Treat `event_fraction` as lower priority unless a later split shows it generalizes better.",
        "- Add stratified split and held-out-energy evaluation before treating this as a final physics result.",
        "",
    ]

    with open(path, "w") as file:
        file.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the pair-event benchmark suite and write a stage report."
    )
    parser.add_argument("datafiles", nargs="*", default=DEFAULT_DATAFILES)
    parser.add_argument("--outdir", default="benchmarks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced", "f1", "recall95"], default="balanced")
    parser.add_argument("--skip-tune", action="store_true", help="Only run the baseline and untuned configs.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(SEED)
    features, labels = load_events(args.datafiles)
    indices = split_indices(len(labels))
    pos_weight = train_pos_weight(labels, indices[0])

    split_summary = split_counts(labels, indices)
    feature_summary = summarize_feature_statistics(features, labels)

    results = []
    mlp_zero, mlp_tuned = train_mlp_baseline(
        features,
        labels,
        indices,
        pos_weight,
        args.epochs,
        args.threshold_metric,
    )
    results.extend([mlp_zero, mlp_tuned])

    cnn_zero, cnn_tuned = train_original_2d_cnn(
        features,
        labels,
        indices,
        pos_weight,
        args.epochs,
        args.threshold_metric,
    )
    results.extend([cnn_zero, cnn_tuned])

    hybrid_main_config = RunConfig("log_block", 1e-3, 1e-4, 0.25, 16, 8, 64, args.epochs)
    torch.manual_seed(SEED)
    results.append(
        run_configured_experiment(
            "hybrid",
            features,
            labels,
            indices,
            pos_weight,
            hybrid_main_config,
            args.threshold_metric,
            "Main hybrid",
            tuned=True,
        )
    )

    if not args.skip_tune:
        for model_type in ["hybrid", "2d"]:
            for config in tune_configs(args.epochs):
                torch.manual_seed(SEED)
                results.append(
                    run_configured_experiment(
                        model_type,
                        features,
                        labels,
                        indices,
                        pos_weight,
                        config,
                        args.threshold_metric,
                        f"Tune {model_type}",
                        tuned=True,
                    )
                )

    payload = {
        "datafiles": args.datafiles,
        "seed": SEED,
        "threshold_metric": args.threshold_metric,
        "split_summary": split_summary,
        "feature_summary": feature_summary,
        "results": results,
    }

    json_path = os.path.join(args.outdir, "pair_detection_benchmark.json")
    csv_path = os.path.join(args.outdir, "pair_detection_benchmark.csv")
    report_path = os.path.join(args.outdir, "pair_detection_stage_report.md")

    write_json(json_path, payload)
    write_csv(csv_path, results)
    write_markdown_report(report_path, args.datafiles, SEED, split_summary, feature_summary, results)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
