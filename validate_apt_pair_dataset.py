"""Validate an APT WLS-only dataset before starting a production campaign."""

import argparse
import json
from pathlib import Path

import numpy as np

from build_apt_pair_dataset import read_gun_event_ids, read_pair_truth


EXPECTED_FEATURE_ORDER = [
    "WLS_Fast_x",
    "WLS_Fast_y",
    "WLS_Slow_x",
    "WLS_Slow_y",
]


def read_effective_config(path):
    values = {}
    with path.open() as source:
        for line in source:
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = parts[1]
    return values


def validate(metadata_path, config_log=None, min_class_count=1):
    metadata_path = metadata_path.resolve()
    with metadata_path.open() as source:
        metadata = json.load(source)
    if metadata.get("format") != "apt_pair_dataset_v1":
        raise ValueError("Not an APT pair dataset metadata file")

    geometry = metadata["geometry"]
    expected_geometry = {
        "feature_order": EXPECTED_FEATURE_ORDER,
        "layers": 20,
        "channels_per_direction": 1492,
        "feature_shape": [4, 20, 1492],
        "features_per_event": 119360,
        "wls_fibers_per_channel": 1,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            raise ValueError(f"Unexpected geometry {key}: {geometry.get(key)!r} != {expected!r}")

    base = metadata_path.parent
    arrays = metadata["arrays"]
    features = np.load(base / arrays["features"], mmap_mode="r")
    labels = np.load(base / arrays["labels"], mmap_mode="r")
    event_ids = np.load(base / arrays["event_ids"], mmap_mode="r")
    energies = np.load(base / arrays["energy_mev"], mmap_mode="r")
    lengths = {len(features), len(labels), len(event_ids), len(energies)}
    if len(lengths) != 1 or features.shape[1:] != (4, 20, 1492):
        raise ValueError("Array lengths or feature geometry are inconsistent")
    if not np.isfinite(features).all() or np.any(features < 0):
        raise ValueError("Features contain a negative or non-finite WLS signal")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Smoke dataset must contain both labels")
    pair_count = int(labels.sum())
    nonpair_count = int(len(labels) - pair_count)
    if min(pair_count, nonpair_count) < min_class_count:
        raise ValueError(
            f"Minimum class count {min_class_count} not met: pair={pair_count}, non-pair={nonpair_count}"
        )
    if not np.all(energies == metadata["energy_mev"]):
        raise ValueError("Energy array does not match metadata")

    gun_path = Path(metadata["sources"]["gun_truth"])
    csi_path = Path(metadata["sources"]["csi_truth"])
    gun_ids = read_gun_event_ids(gun_path)
    pair_ids, truth_ids, _ = read_pair_truth(csi_path, gun_ids)
    expected_labels = np.asarray([event_id in pair_ids for event_id in event_ids], dtype=np.uint8)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Stored labels disagree with CsI creator-process truth")
    if not set(event_ids.tolist()).issubset(truth_ids):
        raise ValueError("A stored event is missing CsI truth")

    if config_log:
        effective = read_effective_config(config_log.resolve())
        required = {
            "device": "apt",
            "NTKRlayers": "20",
            "WLSFibersPerChannel": "1",
            "background": "0",
            "pileup": "0",
            "GunPart": "gamma",
            "GunSpecOption": "0",
            "fiber_err_model": "complete",
            "digitizer_source": "geant_output",
            "digitizer_output": "none",
            "load_nn_models": "0",
            "use_nn": "0",
        }
        mismatches = {
            key: (effective.get(key), expected)
            for key, expected in required.items()
            if effective.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Effective config mismatch: {mismatches}")
        expected_gev = float(metadata["energy_mev"]) / 1000.0
        for key in ["gun_eng", "Emin", "Emax"]:
            if key not in effective or float(effective[key]) != expected_gev:
                raise ValueError(
                    f"Effective config {key}={effective.get(key)!r}, expected {expected_gev} GeV"
                )
        n_events = int(effective["n_events"])
        n_threads = int(effective["n_threads"])
        if n_events % n_threads:
            raise ValueError(f"n_events={n_events} is not divisible by n_threads={n_threads}")

    print(
        f"Validated {metadata_path}: {len(labels)} WLS-only APT events, "
        f"pair={pair_count}, non-pair={nonpair_count}, shape={features.shape}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--config-log", type=Path)
    parser.add_argument("--min-class-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    validate(arguments.metadata, arguments.config_log, arguments.min_class_count)
