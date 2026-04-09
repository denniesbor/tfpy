# tfgic

Geomagnetically Induced Current (GIC) prediction pipeline for power transmission networks using transfer function and physics-informed neural network models.

## Overview

This project implements multiple transfer function and deep learning models to predict GIC in power grids during geomagnetic storms, using TVA (Tennessee Valley Authority) magnetometer and GIC monitoring data from the May 2024 storm event.

## Models

| Key | Model | Method |
|-----|-------|--------|
| `3_td` | Model 3 TD | Time-domain transfer function via MT-derived E-fields |
| `3_fd` | Model 3 FD | Frequency-domain transfer function via MT-derived E-fields |
| `4` | Model 4 | Frequency-domain B-field to GIC transfer function |
| `heyns_td` | Heyns TD | Ensemble with temporal subsampling |
| `heyns_ens` | Heyns FD | Sliding-window ensemble transfer function |
| `5a` | CNN | 1-D convolutional ensemble |
| `5b` | GRU | Bidirectional GRU ensemble with attention |
| PINN | GICPINN | Physics-informed GRU with impedance layer |

## Project Structure

```
config/          - Logging, paths, and device configuration
scripts/         - Data loading, transfer function, neural network, and PINN models
utils/           - Geographic and signal processing utilities
viz/             - Plotting and figure generation
data/            - Runtime data (not tracked — see Data below)
run_regression.py  - Run all transfer function and neural network models
run_pinn.py        - Train and evaluate GICPINN
run_viz.py         - Generate all figures
```

## Data

Data is not included in this repository. Contact [Dennies Bor](mailto:dbor@gmu.edu) to request the zipped data archive. Once received:

```bash
unzip tfgic_data.zip -d data/
```

Expected contents:

```
data/
├── tva_gic.nc
├── tva_magnetometer_data.nc
├── mt_pickle.pkl
├── trans_lines_within_FERC_filtered.pkl
└── TL/
    └── Electric__Power_Transmission_Lines.shp
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/denniesbor/tfgic.git
cd tfgic

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate tf-gic

# 3. Install the package
pip install -e . --no-deps
```

## Usage

```bash
# Run all regression models
run-regression

# Train and evaluate GICPINN
run-pinn

# Generate all figures
run-viz
```

## Performance

Models evaluated using Prediction Efficiency (PE), RMSE, and correlation on test data (May 11 2024, 06:00–18:00 UTC).

## Author

Dennies Bor