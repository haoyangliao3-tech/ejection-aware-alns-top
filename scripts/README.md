# Reproduction scripts

Install the reported environment and arrange both benchmark families as described in `benchmarks/README.md`:

```bash
python -m pip install -e ".[reproduction,dev]"
```

All reported batches use seeds `0 1 2 3` and `--workers 2`. Run each command separately for `--dataset dang` and `--dataset chao`; use the corresponding output directory and the fixed-iteration reference JSON generated for that dataset.

## Fixed-iteration Ejection ON/OFF

```bash
python scripts/experiments/run_ejection_benchmark.py \
  --dataset dang --iterations 2500 --seeds 0 1 2 3 \
  --workers 2 --mode compare --output-dir outputs/dang_fixed_ejection
```

For Chao, replace `dang` with `chao` and use a distinct output directory.

## Fixed-iteration GRASP/ILS/VNS

```bash
python scripts/experiments/run_horizontal_baselines.py \
  --benchmark-root benchmarks --dataset dang --expected-instance-count 82 \
  --algorithms grasp ils vns --iterations 2500 \
  --seeds 0 1 2 3 --workers 2 --output-dir outputs/dang_fixed_baselines
```

For Chao, use `--dataset chao --expected-instance-count 157`.

## Symmetric wall-clock Ejection ON/OFF

Run once with `--ejection on` and once with `--ejection off`:

```bash
python scripts/experiments/run_ejection_off_time_matched.py \
  --reference-json outputs/dang_fixed_ejection/REFERENCE.json \
  --benchmark-root benchmarks --dataset dang --expected-instance-count 82 \
  --reference-iterations 2500 --ejection on --wall-clock-cooling \
  --seeds 0 1 2 3 --workers 2 --output-dir outputs/dang_time_on
```

The Chao commands use the Chao reference JSON, `--dataset chao`, and `--expected-instance-count 157`.

## Runtime-budget GRASP/ILS/VNS/PyVRP

```bash
python scripts/experiments/run_time_matched_baselines.py \
  --reference-json outputs/dang_fixed_ejection/REFERENCE.json \
  --benchmark-root benchmarks --dataset dang --expected-instance-count 82 \
  --budget-mode instance_seed_mean --algorithms grasp ils vns pyvrp \
  --reference-iterations 2500 --seeds 0 1 2 3 --workers 2 \
  --output-dir outputs/dang_runtime_baselines
```

The Chao command again uses its own reference JSON and the Chao dataset flags. PyVRP is included only in the time-budget experiment because its native stopping rule is time based.

## Focused studies

```bash
python scripts/experiments/run_sensitivity_history.py --dataset dang --workers 2
python scripts/experiments/run_review_mechanism_experiments.py --dataset dang --workers 2 --iterations 2500
```

Repeat with `--dataset chao` and separate output directories. The scripts use the predefined nine-instance selections documented in their manifests.

## Data regeneration and QA

- `analysis/build_public_data_bundle.py`: rebuilds the normalized package from completed Dang and Chao logs.
- `analysis/export_result_tables.py`: deterministic JSON-to-CSV export.
- `analysis/build_results_workbook.mjs`: formatted Excel export using `@oai/artifact-tool`.
- `analysis/validate_reported_results.py`: checks counts, four-seed coverage, workers, feasibility, budget compliance, gaps, and aggregates.
- `analysis/build_benchmark_manifest.py`: recomputes the 239-file checksum manifest.
- `figures/plot_unified_w2.py` and `figures/plot_chapter5_additional.py`: regenerate the current two-benchmark result figures from `data/figure_source`.

Wall-clock results are hardware-sensitive. Reproduction should use the same concurrency boundary and an otherwise quiet machine.
