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
| `train_threshold_norm_tune.py` | Runs normalization, threshold, and small hyperparameter sweeps for 2D/hybrid CNNs. |
| `benchmark_pair_detection.py` | Runs the staged benchmark suite and writes JSON, CSV, and Markdown reports. |
| `build_apt_pair_dataset.py` | Joins native digitizer WLS rows to Geant pair truth and writes memory-mapped arrays. |
| `build_apt_manifest.py` | Combines independent APT seed-runs without copying feature arrays. |
| `validate_apt_pair_dataset.py` | Checks APT geometry, labels, truth, signals, and effective configuration. |
| `train_apt_pair_models.py` | Trains WLS-only APT CNN/hybrid models with run-isolated evaluation. |

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

## Running the Stage Benchmark Report

```bash
python3 benchmark_pair_detection.py
```

Use a Python environment with PyTorch installed. On the current local machine,
`/usr/local/bin/python3.11` has been verified to work:

```bash
/usr/local/bin/python3.11 benchmark_pair_detection.py
```

By default, this runs the staged benchmark on:

```text
classifier_data_5MeV.txt classifier_data_10MeV.txt classifier_data_50MeV.txt
```

The benchmark trains the original MLP, the original 2D CNN, the main `log_block`
hybrid CNN, and the small tuned grids for hybrid and 2D CNN models. It writes:

- `benchmarks/pair_detection_benchmark.json`
- `benchmarks/pair_detection_benchmark.csv`
- `benchmarks/pair_detection_stage_report.md`

Use `--skip-tune` for a faster smoke run that only trains the baseline, original
2D CNN, and main hybrid CNN:

```bash
python3 benchmark_pair_detection.py --skip-tune
```

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

## APT WLS-only workflow

The APT path is separate from the legacy four-layer ADAPT parser above. Convert
each energy/seed run after the pipeline's Geant and digitizer passes:

```bash
python build_apt_pair_dataset.py \
  --digitizer /data/apt/5MeV/seed_5005/digitizer_final.txt \
  --csi-truth /data/apt/5MeV/seed_5005/source_particle/CsIout_tmp.dat \
  --gun-truth /data/apt/5MeV/seed_5005/source_particle/GUNout_tmp.dat \
  --output-prefix /data/apt/datasets/5MeV_seed_5005 \
  --energy-mev 5 --seed 5005 --run-id 5MeV_seed_5005 \
  --pipeline-config /path/to/apt_pipeline/config/pair_detection/apt_pair_5mev.config \
  --effective-config-log /data/apt/5MeV/seed_5005/digitizer.log \
  --pipeline-repo /path/to/apt_pipeline \
  --exclude-missing-csi-truth

python validate_apt_pair_dataset.py \
  /data/apt/datasets/5MeV_seed_5005.metadata.json \
  --config-log /data/apt/5MeV/seed_5005/digitizer.log
```

Generate at least three independent seed-runs per energy. This allows the
trainer to keep entire simulation runs in one partition while retaining all
three energies in train, validation, and test. Build one manifest over all run
metadata files, then train:

```bash
python build_apt_manifest.py /data/apt/datasets/*.metadata.json \
  --output /data/apt/datasets/manifest.json

python train_apt_pair_models.py /data/apt/datasets/manifest.json \
  --outdir apt_benchmarks
```

The tensor shape is `[events, 4, 20, 1492]`, ordered as WLS-fast X/Y followed
by WLS-slow X/Y. Labels are pair when any CsI truth hit has creator process
`conv`. Tracker, gun, edge-detector, and calorimeter rows never enter the tensor.
Digitized WLS events without any CsI truth row are excluded without receiving a
label when `--exclude-missing-csi-truth` is explicitly supplied; the number
excluded is recorded in metadata.
The trainer requires 500 events of each class per energy by default; append
another 6,400-event seed-run when that threshold is not met. Override
`--min-class-count-per-energy` only for a smoke test.

The historical ADAPT report remains a useful reference, but it used WLS,
edge-detector, and calorimeter inputs. It is therefore not an apples-to-apples
comparison with these WLS-only APT models.

### Bounded 10/15 MeV exploratory campaign

The simulation configuration and generation scripts live in `~/apt_pipeline`;
the conversion, validation, training, and reporting code lives in this
repository. The current exploratory campaign generates ten independent
2,560-photon runs at each energy under:

```text
/ssd_data/boran.y/apt_pair_detection_exploratory_v2
```

For a new campaign, choose a new output directory so previous data is not
overwritten. From the pipeline repository, run the smoke tests followed by all
twenty production runs:

```bash
cd ~/apt_pipeline
scripts/pair_detection/run_apt_pair_campaign.sh \
  --output-root /ssd_data/boran.y/apt_pair_detection_exploratory_v3
```

The campaign runner uses 64-event smoke tests, seeds 21000-21009 at 10 MeV,
seeds 21500-21509 at 15 MeV, and a 20 GiB storage guard. Completed runs are
skipped when resuming. It writes raw truth, digitizer data, logs, effective
configuration output, and `raw/SHA256SUMS`.

After simulation completes, create and validate all twenty memory-mapped run
shards. Pass the same campaign root used above:

```bash
cd ~/adapt_pair_detection
python3 prepare_apt_campaign.py \
  --root /ssd_data/boran.y/apt_pair_detection_exploratory_v3 \
  --pipeline-repo ~/apt_pipeline
```

The manifest assigns seed endings 0-7 to training, 8 to validation, and 9 to
testing. Train the initial CNN without a hyperparameter sweep:

```bash
./run_apt_exploratory_training.sh \
  /ssd_data/boran.y/apt_pair_detection_exploratory_v3
```

Use a dedicated environment on machines without an existing PyTorch install:

```bash
python3 -m venv --without-pip .venv-apt
python_version=$(.venv-apt/bin/python -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pip3 install --target ".venv-apt/lib/python${python_version}/site-packages" \
  -r requirements-apt.txt
```

The explicit target install also works on the APT server, where the system
Python does not provide `ensurepip`.

The training wrapper removes the server's C++-only `/usr/lib/libtorch` library
path before importing the pinned Python wheel; it does not alter the Geant or
APT pipeline environment.

The report includes usable-event efficiency, per-energy confusion metrics, and
95% uncertainty intervals. The historical ADAPT score is included only as a
contextual reference because its inputs, energies, and split protocol differ.
Results are written to `<campaign-root>/results/`, including the model
checkpoint, JSON metrics, Markdown report, and checksums.

Before committing future model or preprocessing changes, run:

```bash
env -u LD_LIBRARY_PATH .venv-apt/bin/python -B test_apt_pair_pipeline.py
```

To change model settings without modifying the wrapper, invoke the trainer
directly. For example:

```bash
env -u LD_LIBRARY_PATH .venv-apt/bin/python train_apt_pair_models.py \
  /ssd_data/boran.y/apt_pair_detection_exploratory_v3/datasets/manifest.json \
  --outdir /ssd_data/boran.y/apt_pair_detection_exploratory_v3/results_trial \
  --models cnn --epochs 10 --patience 3 --device cpu
```

Use a new results directory for experimental models so the baseline checkpoint
and report remain intact. The bounded preprocessing driver intentionally
expects the fixed 10/15 MeV seed scheme above; update `ENERGY_SEEDS` in
`prepare_apt_campaign.py` if a future campaign changes those energies or seeds.
