# Computational result data

This directory contains the corrected results supporting the current manuscript. Every main-study observation is an algorithm-instance-seed run. All reported batches used `workers=2`, meaning that at most two independent runs were active concurrently and each run used one solver process.

## Coverage

- Dang benchmark: 82 instances.
- Chao benchmark, Sets 4-7: 157 instances.
- Four seeds (`0-3`) for all main fixed-iteration and equal-time studies.
- Two seeds (`0-1`) for the nine-instance-per-benchmark sensitivity and mechanism studies.

Dang and Chao aggregates are reported separately. The package does not pool gaps across benchmark families.

## Files

- `results/normalized_results.json`: canonical machine-readable package.
- `results/Detailed_Instance_Results.xlsx`: formatted workbook with the same records, formula-based gap checks, QA summaries, and statistics.
- `results/csv/benchmark_bks.csv`: 239 BKS references.
- `results/csv/fixed_iteration.csv`: 4,780 runs; Ejection ON/OFF, GRASP, ILS, and VNS at 2,500 iterations.
- `results/csv/symmetric_wall_clock.csv`: 1,912 Ejection ON/OFF runs under identical instance-specific hard-wall budgets.
- `results/csv/runtime_budget_baselines.csv`: 3,824 time-stopped GRASP, ILS, VNS, and PyVRP runs under the same budgets.
- `results/csv/runtime_budgets.csv`: the 239 instance-specific Ejection-ON-derived budgets.
- `results/csv/sensitivity.csv`: 252 parameter-sensitivity runs.
- `results/csv/focused_mechanism.csv`: 252 focused runs with run-level mechanism counters.
- `results/csv/qa_summary.csv`: 22 benchmark/protocol/algorithm aggregate rows.
- `results/csv/reported_statistics.csv`: 36 paired comparison records.
- `results/csv/component_statistics.csv`: 12 instance-level focused-study comparison records.
- `audit/mechanism_cases_dang.json` and `audit/mechanism_cases_chao.json`: event-level bounded-repair audit cases.
- `audit/selected_cases_dang.json`: the selected Dang route-level audit examples.
- `figure_source/`: current Dang/Chao instance summaries and the two-benchmark summary JSON.
- `benchmark_manifest.csv`: BKS, byte count, SHA-256 checksum, source DOI, and expected path for all 239 external benchmark files.

## Core variables

| Variable | Meaning |
|---|---|
| `Benchmark` | `Dang` or `Chao`. |
| `Algorithm` | Algorithm or Ejection ON/OFF arm. |
| `Instance` | Benchmark instance stem. |
| `Seed` | Pseudorandom seed. |
| `Reward` | Final collected TOP reward; larger is better. |
| `BKS` | Published best-known reward used as the reference. |
| `BKS_Gap_Percent` | `100 * (BKS - Reward) / BKS`. |
| `Runtime_Seconds` | Wall-clock runtime per seed in seconds. |
| `Completed_Iterations` | Completed algorithm-specific search cycles. |
| `Budget_Seconds` | Instance-specific hard-wall budget derived from the fixed-iteration Ejection ON runs. |
| `Within_Budget` | Whether the recorded runtime stayed within the assigned budget. |
| `Feasible` | Result of the common post-run feasibility check. |
| `Batch_Workers` | Maximum number of concurrently active independent runs; always `2` in this release. |
| `Source_Record` | Relative provenance key for the source experiment block. |

## Experimental separation

`fixed_iteration` is the 2,500-iteration comparison. `symmetric_wall_clock` contains only the newly time-stopped Ejection ON/OFF arms. `runtime_budget_baselines` contains only time-stopped external comparators. Equal-time summaries combine the latter two tables; fixed-iteration reference records are never mixed into them.

The JSON package was generated from the completed unified experiment directories without changing rewards, runtimes, iteration counts, or feasibility states. CSV files are deterministic exports. Rebuild and validate them with:

```bash
python scripts/analysis/export_result_tables.py
python scripts/analysis/validate_reported_results.py
```

Study-generated data are available under CC BY 4.0. Benchmark instances are not relicensed or redistributed; see `benchmarks/README.md` and `THIRD_PARTY.md`.
