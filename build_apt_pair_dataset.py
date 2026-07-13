"""Convert native apt_pipeline output into a lazy, geometry-aware APT dataset."""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FORMAT_VERSION = "apt_pair_dataset_v1"
WLS_TYPES = {"WLS_Fast": 0, "WLS_Slow": 2}
AXIS_OFFSETS = {"x": 0, "y": 1}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo):
    if repo is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def native_wls_rows(path, layers, channels):
    """Yield (event, map, layer, channel, signal) from native digitizer output."""
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            parts = line.split()
            if not parts or parts[0] == "eventid":
                continue
            if len(parts) != 10:
                raise ValueError(
                    f"{path}:{line_number}: expected 10 native digitizer columns, got {len(parts)}"
                )
            row_type = parts[2]
            if row_type not in WLS_TYPES:
                continue
            axis = parts[3]
            if axis not in AXIS_OFFSETS:
                raise ValueError(f"{path}:{line_number}: invalid WLS axis {axis!r}")

            event_id = int(parts[0])
            layer = int(parts[1])
            channel_float = float(parts[4])
            channel = int(channel_float)
            signal = float(parts[8])
            if channel_float != channel:
                raise ValueError(f"{path}:{line_number}: non-integral channel ID {channel_float}")
            if not 0 <= layer < layers:
                raise ValueError(f"{path}:{line_number}: layer {layer} is outside [0, {layers})")
            if not 0 <= channel < channels:
                raise ValueError(
                    f"{path}:{line_number}: channel {channel} is outside [0, {channels})"
                )
            if not math.isfinite(signal) or signal < 0:
                raise ValueError(f"{path}:{line_number}: invalid signal {signal}")

            map_index = WLS_TYPES[row_type] + AXIS_OFFSETS[axis]
            yield event_id, map_index, layer, channel, signal


def read_gun_event_ids(path):
    event_ids = set()
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            parts = line.split()
            if not parts:
                continue
            try:
                event_id = int(parts[0])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid GUN event ID") from error
            if event_id in event_ids:
                raise ValueError(f"{path}:{line_number}: duplicate GUN event ID {event_id}")
            event_ids.add(event_id)
    if not event_ids:
        raise ValueError(f"{path}: no GUN truth records found")
    return event_ids


def read_pair_truth(path, gun_event_ids):
    pair_ids = set()
    truth_event_ids = set()
    truth_rows = 0
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 17:
                raise ValueError(f"{path}:{line_number}: malformed CsI truth row")
            event_id = int(parts[0])
            if event_id not in gun_event_ids:
                raise ValueError(f"{path}:{line_number}: event {event_id} is absent from GUN truth")
            truth_rows += 1
            truth_event_ids.add(event_id)
            if parts[-1] == "conv":
                pair_ids.add(event_id)
    if not truth_event_ids:
        raise ValueError(f"{path}: no CsI truth records found")
    return pair_ids, truth_event_ids, truth_rows


def output_paths(prefix):
    return {
        "features": prefix.with_suffix(".features.npy"),
        "labels": prefix.with_suffix(".labels.npy"),
        "event_ids": prefix.with_suffix(".event_ids.npy"),
        "energy_mev": prefix.with_suffix(".energy_mev.npy"),
        "metadata": prefix.with_suffix(".metadata.json"),
    }


def convert(args):
    digitizer = args.digitizer.resolve()
    csi_truth = args.csi_truth.resolve()
    gun_truth = args.gun_truth.resolve()
    pipeline_config = args.pipeline_config.resolve() if args.pipeline_config else None
    effective_config_log = (
        args.effective_config_log.resolve() if args.effective_config_log else None
    )
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for required in [digitizer, csi_truth, gun_truth]:
        if not required.is_file():
            raise FileNotFoundError(required)

    ordered_event_ids = []
    seen_events = set()
    wls_row_count = 0
    for event_id, *_ in native_wls_rows(digitizer, args.layers, args.channels):
        wls_row_count += 1
        if event_id not in seen_events:
            seen_events.add(event_id)
            ordered_event_ids.append(event_id)
    if not ordered_event_ids:
        raise ValueError(f"{digitizer}: no WLS rows found")

    gun_event_ids = read_gun_event_ids(gun_truth)
    missing_gun = seen_events - gun_event_ids
    if missing_gun:
        sample = sorted(missing_gun)[:10]
        raise ValueError(f"Digitized event IDs missing from GUN truth: {sample}")

    pair_ids, truth_event_ids, csi_truth_rows = read_pair_truth(csi_truth, gun_event_ids)
    missing_csi = seen_events - truth_event_ids
    if missing_csi and not args.exclude_missing_csi_truth:
        sample = sorted(missing_csi)[:10]
        raise ValueError(
            f"Digitized WLS event IDs missing from CsI truth: {sample}; "
            "use --exclude-missing-csi-truth to omit them without assigning a label"
        )
    if missing_csi:
        ordered_event_ids = [
            event_id for event_id in ordered_event_ids if event_id in truth_event_ids
        ]
        seen_events = set(ordered_event_ids)
        if not ordered_event_ids:
            raise ValueError("No digitized WLS events have CsI truth")

    event_ids = np.asarray(ordered_event_ids, dtype=np.int64)
    labels = np.asarray([event_id in pair_ids for event_id in ordered_event_ids], dtype=np.uint8)
    energies = np.full(len(event_ids), args.energy_mev, dtype=np.float32)
    pair_count = int(labels.sum())
    nonpair_count = int(len(labels) - pair_count)
    if not args.allow_single_class and (pair_count == 0 or nonpair_count == 0):
        raise ValueError(
            f"Dataset must contain both classes; found pair={pair_count}, non-pair={nonpair_count}"
        )
    if min(pair_count, nonpair_count) < args.min_class_count:
        raise ValueError(
            f"Minimum class count {args.min_class_count} not met: "
            f"pair={pair_count}, non-pair={nonpair_count}"
        )

    event_index = {event_id: index for index, event_id in enumerate(ordered_event_ids)}
    paths = output_paths(prefix)
    shape = (len(ordered_event_ids), 4, args.layers, args.channels)
    features = np.lib.format.open_memmap(
        paths["features"], mode="w+", dtype=np.float32, shape=shape
    )
    features[:] = 0
    included_wls_rows = 0
    for event_id, map_index, layer, channel, signal in native_wls_rows(
        digitizer, args.layers, args.channels
    ):
        if event_id not in event_index:
            continue
        features[event_index[event_id], map_index, layer, channel] = signal
        included_wls_rows += 1
    features.flush()
    del features

    np.save(paths["event_ids"], event_ids)
    np.save(paths["labels"], labels)
    np.save(paths["energy_mev"], energies)

    metadata = {
        "format": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id or prefix.name,
        "instrument": "apt",
        "energy_mev": args.energy_mev,
        "random_seed": args.seed,
        "geometry": {
            "feature_order": ["WLS_Fast_x", "WLS_Fast_y", "WLS_Slow_x", "WLS_Slow_y"],
            "layers": args.layers,
            "channels_per_direction": args.channels,
            "feature_shape": list(shape[1:]),
            "features_per_event": int(np.prod(shape[1:])),
            "wls_fibers_per_channel": 1,
        },
        "truth": {
            "pair_rule": "any CsIout row with creator process 'conv'",
            "csi_truth_rows": csi_truth_rows,
            "csi_truth_events": len(truth_event_ids),
            "missing_csi_policy": (
                "excluded_without_label" if missing_csi else "no_missing_events"
            ),
        },
        "counts": {
            "events": len(event_ids),
            "pair": pair_count,
            "nonpair": nonpair_count,
            "wls_rows": included_wls_rows,
            "source_wls_rows": wls_row_count,
            "excluded_missing_csi_truth": len(missing_csi),
        },
        "arrays": {key: path.name for key, path in paths.items() if key != "metadata"},
        "sources": {
            "digitizer": str(digitizer),
            "csi_truth": str(csi_truth),
            "gun_truth": str(gun_truth),
            "pipeline_config": str(pipeline_config) if pipeline_config else None,
            "pipeline_config_sha256": file_sha256(pipeline_config) if pipeline_config else None,
            "effective_config_log": (
                str(effective_config_log) if effective_config_log else None
            ),
            "effective_config_log_sha256": (
                file_sha256(effective_config_log) if effective_config_log else None
            ),
            "pipeline_commit": args.pipeline_commit or git_commit(args.pipeline_repo),
        },
    }
    with paths["metadata"].open("w") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(
        f"Wrote {paths['metadata']} with {len(labels)} events "
        f"({pair_count} pair, {nonpair_count} non-pair)"
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join APT Geant truth with native digitizer WLS output."
    )
    parser.add_argument("--digitizer", type=Path, required=True)
    parser.add_argument("--csi-truth", type=Path, required=True)
    parser.add_argument("--gun-truth", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--energy-mev", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--effective-config-log", type=Path)
    parser.add_argument("--pipeline-repo", type=Path)
    parser.add_argument("--pipeline-commit")
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--channels", type=int, default=1492)
    parser.add_argument("--min-class-count", type=int, default=1)
    parser.add_argument("--allow-single-class", action="store_true")
    parser.add_argument(
        "--exclude-missing-csi-truth",
        action="store_true",
        help="Explicitly omit digitized WLS events that have no CsI truth row.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
