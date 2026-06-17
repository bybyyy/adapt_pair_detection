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

