# Erdos Graph Project

This repository studies Erdős-Rényi random graphs across six phases:

- Phase 2: algorithmic efficiency and complexity comparisons
- Phase 3: structural simulations and phase-transition plots
- Phase 4: resilience and epidemic dynamics
- Phase 5: Manim evolution video
- Phase 6: real-world null-model rejection and synthetic brain-network extension

## Files

- `fast_er.py`: Batagelj-Brandes `G(n, p)` generator
- `naive_er.py`: baseline `O(n^2)` generator
- `plot1.py` to `plot12_er_evolution_video.py`: simulation, benchmarking, and video scripts
- `plot13_real_world_mapping.py`: S&P 500 correlation-network null-model analysis
- `plot14_brain_connectivity.py`: synthetic connectome extension

## Setup

Create and activate a virtual environment, then install the Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Examples:

```bash
python3 plot1.py
python3 plot7_phase_transition_curve.py
python3 plot10_network_resilience.py
python3 plot13_real_world_mapping.py
```

Render the Manim animation with:

```bash
manim -qm plot12_er_evolution_video.py Plot12ErdosRenyiEvolution
```

## Phase 6 Data

`plot13_real_world_mapping.py` now prefers a local CSV cache before attempting a live download.

- Default cache path: `data/sp500_adj_close.csv`
- Override with: `ERDOS_PRICE_DATA=/path/to/file.csv`

The CSV should contain adjusted close prices with:

- a date index in the first column
- ticker symbols as column names

If no cached CSV is present, the script will try to download the data with `yfinance` and save the result to the cache path for future runs.

## Outputs

Generated figures are written to the repo root.
