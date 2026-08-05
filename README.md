# Ejection-Aware ALNS for the Team Orienteering Problem

[![Tests](https://github.com/haoyangliao3-tech/ejection-aware-alns-top/actions/workflows/tests.yml/badge.svg)](https://github.com/haoyangliao3-tech/ejection-aware-alns-top/actions/workflows/tests.yml)

This repository provides a reproducible Python implementation of Ejection-Aware Adaptive Large Neighborhood Search (ALNS) for the Team Orienteering Problem (TOP). The implementation focuses on the repair stage: customers rejected by direct-feasibility filtering remain eligible for a bounded ejection-and-rearrangement move when that move produces a feasible positive gain.

The code accompanies the manuscript *Ejection-Aware Adaptive Large Neighborhood Search for the Team Orienteering Problem*. The repository contains the proposed solver, all reported comparator implementations or adapters, experiment and analysis scripts, individual algorithm-instance-seed results, figure source data, mechanism audit records, a synthetic example, and unit tests. Manuscript source files remain outside this release.

## Method overview

The solver combines:

- adaptive destroy-operator and node-selection policies;
- attention-inspired repair scoring;
- reconsideration of blocked customers;
- bounded repair-level ejection ranked by removal density;
- bounded 2-opt feasibility recovery; and
- simulated-annealing acceptance.

The manuscript configuration uses the following ejection parameters:

| Parameter | Meaning | Default |
|---|---|---:|
| `K` | maximum ejection attempts | 100 |
| `P` | insertion positions retained per candidate | 3 |
| `C` | maximum ejected customers | 2 |
| `L` | bounded 2-opt passes | 1 |

These values are engineering defaults within the ranges examined in the associated study, not globally optimized parameters.

## Requirements

- Python 3.10 or newer
- No mandatory third-party runtime dependency
- `pytest` only for development and testing
- Optional NumPy/Numba acceleration through the `performance` extra

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/haoyangliao3-tech/ejection-aware-alns-top.git
cd ejection-aware-alns-top
python -m pip install -e ".[dev]"
```

Install optional compiled 2-opt acceleration with:

```bash
python -m pip install -e ".[performance]"
```

Install the exact computational-reproduction dependencies with:

```bash
python -m pip install -e ".[reproduction,dev]"
```

## Quick start

Run the included synthetic instance:

```bash
ejection-aware-alns data/sample/sample_top_instance.txt \
  --iterations 2500 \
  --seed 0 \
  --output outputs/toy_result.json
```

The same runner is available as a Python module:

```bash
python -m top_alns.experiments.runner \
  data/sample/sample_top_instance.txt \
  --iterations 2500 \
  --seed 0
```

Disable Ejection-Aware Repair for a controlled run:

```bash
ejection-aware-alns data/sample/sample_top_instance.txt \
  --iterations 2500 \
  --seed 0 \
  --ejection-off
```

## Python API

```python
from top_alns.alns import ALNSolver
from top_alns.parser import parse_instance

instance = parse_instance("data/sample/sample_top_instance.txt")
solver = ALNSolver(
    max_iterations=2500,
    random_seed=0,
    enable_ejection=True,
    ejection_max_attempts=100,
    ejection_max_positions=3,
    ejection_max_nodes=2,
    ejection_two_opt_passes=1,
)
solution = solver.solve(instance)

print(solution.total_reward)
print([route.node_ids for route in solution.routes])
```

## Instance formats

The compact format begins with:

```text
vehicle_count max_distance depot_id
```

Each following row contains:

```text
node_id x y reward
```

The parser also accepts the public Chao/Dang TOP text format with `n`, `m`, and `tmax` headers. Benchmark collections are not redistributed here. Obtain the Dang instances from their original source, place them under `benchmarks/Dang et al., (2013)/`, and verify them against `data/benchmark_manifest.csv`. See `benchmarks/README.md`.

## Computational results and reproducibility

The repository exposes deterministic random seeds and all solver parameters used by the runners. The `data/results` directory contains every reported algorithm-instance-seed outcome in JSON, CSV, and Excel formats. `data/figure_source` maps the plotted values to manuscript figures, while `data/audit` contains the bounded mechanism case records. Detailed schemas and provenance are documented in `data/README.md`.

Recreate the CSV tables and validate all reported aggregates with:

```bash
python scripts/analysis/export_result_tables.py
python scripts/analysis/validate_reported_results.py
```

Commands for rerunning the fixed-iteration, symmetric wall-clock, runtime-budget, sensitivity, and focused mechanism experiments are documented in `scripts/README.md`.

Exact wall-clock times depend on hardware, operating system, Python version, and system load. Use isolated single-worker execution and a consistent environment when reproducing runtime comparisons.

## Tests

```bash
python -m pytest
```

The continuous-integration workflow runs the test suite on Python 3.10 and 3.12.

## Citation

If this implementation supports your research, cite the associated manuscript. Machine-readable author and software metadata are provided in `CITATION.cff`. Publication metadata will be added after the manuscript is published.

## License

The software is released under the MIT License in `LICENSE`. Data generated by this study are released under CC BY 4.0 as described in `DATA_LICENSE.md`. Third-party software and benchmark sources retain their original licences and terms; see `THIRD_PARTY.md`.
