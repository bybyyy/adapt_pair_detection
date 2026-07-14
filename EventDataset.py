import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TKR_COUNT = 600
# 75 per direction per layer
WLS_FAST_COUNT = 75 * 2 * 4
WLS_SLOW_COUNT = 75 * 2 * 4
# Edge detector, 3 per direction per layer
ED_COUNT = 24
CAL_COUNT = 24
TOTAL_FEATURES = WLS_SLOW_COUNT + WLS_FAST_COUNT + ED_COUNT + CAL_COUNT

class EventDataset(Dataset):
    def __init__(self, file_path):
        
        self.event_inputs, self.event_labels = self.parse_file(file_path)
        self.pair_count = self.event_labels.count(1)
        self.total_count = len(self.event_labels)
        print(f'{self.pair_count} pair events out of {self.total_count} total')

        # python list > np array > pytorch tensor
        # there's probably a better way to do this
        self.features_tensor = torch.tensor(np.array(self.event_inputs), dtype=torch.float32)
        self.labels_tensor = torch.tensor(self.event_labels, dtype=torch.float32)

        # scale?
        # max_vals = self.features_tensor.max(dim=0, keepdim=True)[0]
        # max_vals[max_vals == 0] = 1.0
        # self.features_tensor = self.features_tensor / max_vals


    def parse_file(self, file_path):

        data_inputs = []
        data_labels = []
        curr_row = None
        curr_type = 0

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if (len(parts) == 0):
                    continue
                row_type = parts[0]

                # TODO: convert this to a switch case it would be so so much better
                if 'EVENT' in row_type:
                    # Structure: EVENT [id] [x] [y] [z] [KE (disregard)]
                    if curr_row is not None:
                        data_inputs.append(curr_row.copy())
                        data_labels.append(curr_type)

                    curr_row = np.zeros(TOTAL_FEATURES)
                    curr_type = 1 if 'PAIR' in row_type else 0                            

                # TODO: check that row counting error (directions) is fixed
                elif row_type == 'WLS_Fast':
                    if curr_row is None:  
                        raise ValueError("Couldn't read event from file", file_path)
                    direction = 1 if parts[2] == 'y' else 0
                    component_id = (int(parts[1]) * 75 * 2) + direction * 75 + int(parts[3])
                    signal = float(parts[7])
                    curr_row[component_id] = signal

                elif row_type == 'WLS_Slow':
                    if curr_row is None:  
                        raise ValueError("Couldn't read event from file", file_path)
                    direction = 1 if parts[2] == 'y' else 0
                    component_id = WLS_FAST_COUNT + (int(parts[1]) * 75 * 2) + direction * 75 + int(parts[3])
                    signal = float(parts[7])
                    curr_row[component_id] = signal

                elif row_type == 'Edge_Detector':
                    if curr_row is None:  
                        raise ValueError("Couldn't read event from file", file_path)
                    direction = 1 if parts[2] == 'y' else 0
                    component_id = WLS_FAST_COUNT + WLS_SLOW_COUNT + (int(parts[1]) * 3 * 2) + direction * 3 + int(parts[3])
                    signal = float(parts[7])
                    curr_row[component_id] = signal

                elif row_type == 'Calorimeter':
                    if curr_row is None:  
                        raise ValueError("Couldn't read event from file", file_path)
                    direction = 1 if parts[2] == 'y' else 0
                    component_id = WLS_FAST_COUNT + WLS_SLOW_COUNT + ED_COUNT + (int(parts[1]) * 3 * 2) + direction * 3 + int(parts[3])
                    signal = float(parts[7])
                    curr_row[component_id] = signal

                else:
                    pass

        if curr_row is not None:
            data_inputs.append(curr_row.copy())
            data_labels.append(curr_type)

        return data_inputs, data_labels

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, idx):
        return self.features_tensor[idx], self.labels_tensor[idx]
    
    def get_features(self):
        return self.event_inputs
    
    def get_labels(self):
        return self.event_labels
    
    def get_counts(self):
        return (self.pair_count, self.total_count)


class _AptDatasetShard:
    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path).resolve()
        with self.metadata_path.open() as source:
            self.metadata = json.load(source)
        if self.metadata.get("format") != "apt_pair_dataset_v1":
            raise ValueError(f"Unsupported APT dataset metadata: {self.metadata_path}")

        arrays = self.metadata["arrays"]
        base = self.metadata_path.parent
        self.features = np.load(base / arrays["features"], mmap_mode="r")
        self.labels = np.load(base / arrays["labels"], mmap_mode="r")
        self.event_ids = np.load(base / arrays["event_ids"], mmap_mode="r")
        self.energy_mev = np.load(base / arrays["energy_mev"], mmap_mode="r")
        expected_shape = tuple(self.metadata["geometry"]["feature_shape"])
        if self.features.shape[1:] != expected_shape:
            raise ValueError(
                f"Feature shape mismatch in {self.metadata_path}: "
                f"{self.features.shape[1:]} != {expected_shape}"
            )
        lengths = {len(self.features), len(self.labels), len(self.event_ids), len(self.energy_mev)}
        if len(lengths) != 1:
            raise ValueError(f"Array length mismatch in {self.metadata_path}")

    def __len__(self):
        return len(self.labels)


class AptPairDataset(Dataset):
    """Lazy reader for one APT dataset metadata file or a multi-run manifest."""

    def __init__(self, path):
        path = Path(path).resolve()
        with path.open() as source:
            descriptor = json.load(source)

        if descriptor.get("format") == "apt_pair_dataset_v1":
            metadata_paths = [path]
        elif descriptor.get("format") == "apt_pair_manifest_v1":
            metadata_paths = []
            for entry in descriptor.get("datasets", []):
                candidate = Path(entry)
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                metadata_paths.append(candidate)
            if not metadata_paths:
                raise ValueError(f"APT manifest contains no datasets: {path}")
        else:
            raise ValueError(f"Unsupported dataset descriptor: {path}")

        self.shards = [_AptDatasetShard(item) for item in metadata_paths]
        self.geometry = self.shards[0].metadata["geometry"]
        for shard in self.shards[1:]:
            if shard.metadata["geometry"] != self.geometry:
                raise ValueError(f"Geometry mismatch in {shard.metadata_path}")

        self.cumulative_lengths = []
        total = 0
        for shard in self.shards:
            total += len(shard)
            self.cumulative_lengths.append(total)

        self.labels = np.concatenate([np.asarray(shard.labels) for shard in self.shards])
        self.energy_mev = np.concatenate([np.asarray(shard.energy_mev) for shard in self.shards])
        self.event_ids = np.concatenate([np.asarray(shard.event_ids) for shard in self.shards])
        self.run_ids = np.concatenate(
            [
                np.full(len(shard), shard.metadata["run_id"], dtype=object)
                for shard in self.shards
            ]
        )
        self.random_seeds = np.concatenate(
            [
                np.full(len(shard), shard.metadata["random_seed"], dtype=np.int64)
                for shard in self.shards
            ]
        )
        keys = list(zip(self.run_ids.tolist(), self.event_ids.tolist()))
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate (run_id, event_id) keys in APT manifest")
        self.pair_count = int(self.labels.sum())
        self.total_count = len(self.labels)
        print(f"{self.pair_count} pair events out of {self.total_count} total APT events")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _locate(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_lengths, index)
        start = 0 if shard_index == 0 else self.cumulative_lengths[shard_index - 1]
        return self.shards[shard_index], index - start

    def __getitem__(self, index):
        shard, local_index = self._locate(index)
        features = torch.tensor(shard.features[local_index], dtype=torch.float32)
        label = torch.tensor(float(shard.labels[local_index]), dtype=torch.float32)
        energy = torch.tensor(float(shard.energy_mev[local_index]), dtype=torch.float32)
        return features, label, energy

    def sample_key(self, index):
        return str(self.run_ids[index]), int(self.event_ids[index])

    def get_counts(self):
        return self.pair_count, self.total_count
