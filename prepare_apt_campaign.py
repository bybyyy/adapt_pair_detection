"""Convert and validate the bounded 10/15 MeV APT exploratory campaign."""

import argparse
import hashlib
import json
from pathlib import Path

from build_apt_manifest import build_manifest
from build_apt_pair_dataset import convert, output_paths
from validate_apt_pair_dataset import validate


ENERGY_SEEDS = {
    10: range(21000, 21010),
    15: range(21500, 21510),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    root = args.root.resolve()
    raw_root = root / "raw"
    dataset_root = root / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata_paths = []

    for energy, seeds in ENERGY_SEEDS.items():
        config = args.pipeline_repo / "config" / "pair_detection" / f"apt_pair_{energy}mev.config"
        for seed in seeds:
            run_id = f"{energy}MeV_seed_{seed}"
            run_dir = raw_root / f"{energy}MeV" / f"seed_{seed}"
            prefix = dataset_root / run_id
            metadata_path = output_paths(prefix)["metadata"]
            if metadata_path.exists() and not args.rebuild:
                validate(
                    metadata_path,
                    run_dir / "digitizer.log",
                    min_class_count=0,
                    allow_single_class=True,
                    expected_incident_events=args.incident_events_per_run,
                )
                metadata_paths.append(metadata_path)
                continue

            namespace = argparse.Namespace(
                digitizer=run_dir / "digitizer_final.txt",
                csi_truth=run_dir / "source_particle" / "CsIout_tmp.dat",
                gun_truth=run_dir / "source_particle" / "GUNout_tmp.dat",
                output_prefix=prefix,
                energy_mev=float(energy),
                seed=seed,
                run_id=run_id,
                pipeline_config=config,
                effective_config_log=run_dir / "digitizer.log",
                pipeline_repo=args.pipeline_repo,
                pipeline_commit=None,
                layers=20,
                channels=1492,
                min_class_count=0,
                allow_single_class=True,
                exclude_missing_csi_truth=True,
            )
            metadata = convert(namespace)
            if metadata["counts"]["incident_events"] != args.incident_events_per_run:
                raise ValueError(
                    f"{run_id} has {metadata['counts']['incident_events']} incident events; "
                    f"expected {args.incident_events_per_run}"
                )
            validate(
                metadata_path,
                run_dir / "digitizer.log",
                min_class_count=0,
                allow_single_class=True,
                expected_incident_events=args.incident_events_per_run,
            )
            metadata_paths.append(metadata_path)

    manifest_path = dataset_root / "manifest.json"
    build_manifest(metadata_paths, manifest_path)
    with manifest_path.open() as source:
        manifest = json.load(source)

    for energy in ["10", "15"]:
        counts = manifest["summary_by_energy"][energy]
        expected_incident = 10 * args.incident_events_per_run
        if counts["runs"] != 10 or counts["incident_events"] != expected_incident:
            raise ValueError(f"Incomplete {energy} MeV campaign: {counts}")
        if min(counts["pair"], counts["nonpair"]) < args.min_class_count:
            raise ValueError(
                f"{energy} MeV is underpowered: pair={counts['pair']}, "
                f"nonpair={counts['nonpair']}"
            )

    checksum_path = dataset_root / "SHA256SUMS"
    checksum_files = sorted(
        path for path in dataset_root.iterdir() if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w") as output:
        for path in checksum_files:
            output.write(f"{sha256(path)}  {path.name}\n")
    print(f"Prepared and validated {manifest_path}")
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/ssd_data/boran.y/apt_pair_detection_exploratory_v2"),
    )
    parser.add_argument(
        "--pipeline-repo", type=Path, default=Path("/home/boran.y/apt_pipeline")
    )
    parser.add_argument("--incident-events-per-run", type=int, default=2560)
    parser.add_argument("--min-class-count", type=int, default=500)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
