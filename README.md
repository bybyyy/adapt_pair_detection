# ADAPT Pair Event Detection

This project trains and evaluates machine learning models for identifying pair-production events versus non-pair, Compton-like events in ADAPT detector simulation output. The code parses Geant4-style text output into fixed-length detector feature vectors, then compares a baseline multilayer perceptron with newer CNN-based models that use the detector geometry more directly.

## Project History

This repository started from a teammate's pair-event classification code. The original foundation includes the event parser, dataset wrapper, baseline PyTorch classifier, and the first training entry point:

- `EventDataset.py`
- `PairEventClassifier.py`
- `main.py`

Additional work added on top of that base includes:

- comparison plots for pair and non-pair detector responses in `compare_pair_nonpair_plots.py`
- exploratory plotting updates in `plot_data.py`
- a 2D CNN model and MLP baseline comparison in `train_2d_cnn.py`
- a hybrid CNN model that combines detector-map features with engineered summary features in `train_hybrid_cnn.py`
- `.gitignore` rules to avoid pushing local cache files, generated plots, and large simulation datasets

## Repository Contents

| File | Purpose |
| --- | --- |
| `EventDataset.py` | Parses event text files into PyTorch tensors. |
| `PairEventClassifier.py` | Baseline MLP classifier for binary pair/non-pair classification. |
| `main.py` | Original training script for the baseline classifier. |
| `plot_data.py` | Basic exploratory histograms and event-display plotting. |
| `compare_pair_nonpair_plots.py` | Generates pair/non-pair comparison plots and summary statistics. |
| `train_2d_cnn.py` | Trains the original MLP baseline and a 2D CNN on the same split. |
| `train_hybrid_cnn.py` | Trains the original MLP baseline and a hybrid CNN with engineered features. |

## Data Availability

The training dataset was not pushed to GitHub. The full simulation output can be large, and at least one local file, `data_range.txt`, is larger than GitHub's normal file size limit. Data files should be kept locally or stored separately using an appropriate data-storage method.

The parser expects one or more plain-text simulation output files. These files may include Geant4 header/log text before and between event records. Lines that do not match the expected event or detector record types are ignored.

## Data File Format

Each event starts with an `EVENT` line. The parser labels the event as pair if the first token contains `PAIR`; otherwise it labels the event as non-pair.

```text
EVENT <event_id> <x> <y> <z> <ke_or_unused>
PAIR_EVENT <event_id> <x> <y> <z> <ke_or_unused>
```

Detector hit lines follow the event header. The supported detector row types are:

```text
WLS_Fast <layer> <direction> <strip> <x> <y> <raw_or_aux> <signal>
WLS_Slow <layer> <direction> <strip> <x> <y> <raw_or_aux> <signal>
Edge_Detector <layer> <direction> <cell> <x> <y> <raw_or_aux> <signal>
Calorimeter <layer> <direction> <cell> <x> <y> <raw_or_aux> <signal>
```

Only selected columns are used by `EventDataset.py`:

- `layer`: detector layer index
- `direction`: `x` or `y`
- `strip` or `cell`: channel index within that detector/direction/layer
- `signal`: final numeric column, stored as the feature value

The parsed feature vector has 1,248 values per event:

| Feature block | Shape represented | Count |
| --- | --- | ---: |
| WLS fast | 4 layers x 2 directions x 75 strips | 600 |
| WLS slow | 4 layers x 2 directions x 75 strips | 600 |
| Edge detector | 4 layers x 2 directions x 3 cells | 24 |
| Calorimeter | 4 layers x 2 directions x 3 cells | 24 |
| Total |  | 1,248 |

Missing channels are filled with zero.

## Setup

Create a Python environment and install the main dependencies:

```bash
pip install torch numpy matplotlib
```

The code has been used as simple Python scripts rather than as an installed package.

## Running the Baseline MLP

```bash
python main.py <datafile1> [<datafile2> ...]
```

This loads all provided data files, creates an 80/10/10 train/validation/test split, trains `PairEventClassifier`, and prints a confusion matrix.

## Running the 2D CNN Comparison

```bash
python train_2d_cnn.py <datafile1> [<datafile2> ...]
```

This script trains two models on the same deterministic split:

- `Original MLP`: the baseline classifier using the flattened 1,248-feature vector
- `2D CNN`: a detector-aware CNN that reshapes WLS signals into 2D layer/channel maps and processes edge/calorimeter data as compact detector maps

The script prints validation loss per epoch and final test-set accuracy/confusion matrices.

## Running the Hybrid CNN Comparison

```bash
python train_hybrid_cnn.py <datafile1> [<datafile2> ...]
```

The hybrid CNN uses the same detector-map CNN backbone and appends engineered summary features, including total signal, WLS fast/slow totals, edge/calorimeter totals, active WLS channels, and signal fractions. It compares the hybrid model against the original MLP baseline and prints the accuracy difference.

## Generating Pair vs Non-Pair Plots

```bash
python compare_pair_nonpair_plots.py <datafile1> [<datafile2> ...]
```

By default, plots are saved under `pair_nonpair_plots/`, which is ignored by git. The script generates:

- feature histograms
- feature scatter plots
- standardized mean-difference bars
- average detector response maps
- representative event displays
- a text summary of pair and non-pair feature means

Use `--show` to open plots interactively while also saving them:

```bash
python compare_pair_nonpair_plots.py <datafile1> --show
```

## Notes for Continuing Work

- Keep local data files outside git or under the ignored names in `.gitignore`.
- Commit code and documentation changes separately from generated plots or datasets.
- If large data needs to be shared later, use a separate data release, cloud storage location, or Git LFS after confirming project policy.
