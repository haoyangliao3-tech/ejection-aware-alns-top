# Ejection-Aware ALNS for the Team Orienteering Problem

[![Tests](https://github.com/haoyangliao3-tech/ejection-aware-alns-top/actions/workflows/tests.yml/badge.svg)](https://github.com/haoyangliao3-tech/ejection-aware-alns-top/actions/workflows/tests.yml)

This repository provides a reproducible Python implementation of Ejection-Aware Adaptive Large Neighborhood Search (ALNS) for the Team Orienteering Problem (TOP). The implementation focuses on the repair stage: customers rejected by direct-feasibility filtering remain eligible for a bounded ejection-and-rearrangement move when that move produces a feasible positive gain.

The code accompanies the manuscript *Ejection-Aware Adaptive Large Neighborhood Search for the Team Orienteering Problem*. The repository contains the proposed solver, parsers, a command-line runner, a synthetic example, and unit tests. Manuscript files, raw experimental outputs, and third-party comparator reproductions are intentionally outside this software release.

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

The parser also accepts the public Chao/Dang TOP text format with `n`, `m`, and `tmax` headers. Benchmark collections are not redistributed here. Obtain them from their original sources, retain their original directory separately, and pass an instance path to the runner.

## Reproducibility scope

The repository exposes deterministic random seeds and all solver parameters used by the runner. Each JSON result records the configuration, objective value, routes, feasibility status, runtime, and available search diagnostics.

Exact wall-clock times depend on hardware, operating system, Python version, and system load. Use isolated single-worker execution and a consistent environment when reproducing runtime comparisons.

## Tests

```bash
python -m pytest
```

The continuous-integration workflow runs the test suite on Python 3.10 and 3.12.

## Citation

If this implementation supports your research, cite the associated manuscript. Machine-readable author and software metadata are provided in `CITATION.cff`. Publication metadata will be added after the manuscript is published.

## License

The open-source license is stated in `LICENSE`.
