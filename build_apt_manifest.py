"""Build a manifest over APT dataset shards without copying their arrays."""

import argparse
import json
from pathlib import Path


def build_manifest(metadata_paths, output_path):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = []
    geometry = None
    run_ids = set()
    for metadata_path in metadata_paths:
        metadata_path = metadata_path.resolve()
        with metadata_path.open() as source:
            metadata = json.load(source)
        if metadata.get("format") != "apt_pair_dataset_v1":
            raise ValueError(f"Unsupported metadata format in {metadata_path}")
        current_geometry = metadata["geometry"]
        if geometry is None:
            geometry = current_geometry
        elif current_geometry != geometry:
            raise ValueError(f"Geometry mismatch in {metadata_path}")
        run_id = metadata["run_id"]
        if run_id in run_ids:
            raise ValueError(f"Duplicate run_id {run_id!r}")
        run_ids.add(run_id)
        try:
            relative = metadata_path.relative_to(output_path.parent)
            datasets.append(str(relative))
        except ValueError:
            datasets.append(str(metadata_path))

    manifest = {
        "format": "apt_pair_manifest_v1",
        "datasets": datasets,
        "geometry": geometry,
    }
    with output_path.open("w") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(f"Wrote {output_path} with {len(datasets)} dataset shards")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine APT dataset metadata into a manifest.")
    parser.add_argument("metadata", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_manifest(args.metadata, args.output)

