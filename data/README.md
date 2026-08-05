# Computational result data

## Summary

This directory contains the individual computational outcomes supporting the manuscript *Ejection-Aware Adaptive Large Neighborhood Search for the Team Orienteering Problem*. Each observation is an algorithm-instance-seed run unless otherwise stated. Values are provided in open CSV/JSON formats and in a formatted Excel workbook.

## Files

- `results/normalized_results.json`: canonical machine-readable package containing all result tables and metadata.
- `results/Detailed_Instance_Results.xlsx`: formatted workbook with the same run-level data, live gap checks, QA summaries, and reported statistics.
- `results/csv/benchmark_bks.csv`: BKS reference for each of 82 instances.
- `results/csv/fixed_iteration.csv`: 1,640 runs; Ejection ON, Ejection OFF, GRASP, ILS, and VNS at 2,500 iterations, four seeds, and 82 instances.
- `results/csv/symmetric_wall_clock.csv`: 656 newly run Ejection ON/OFF outcomes under identical instance-specific hard walls and time-based cooling.
- `results/csv/runtime_budget.csv`: 1,968 records comprising the existing Ejection ON fixed-iteration reference and five time-stopped comparators.
- `results/csv/runtime_budgets.csv`: the 82 instance-specific reference budgets used by the time-stopped comparators.
- `results/csv/sensitivity.csv`: 126 runs for the seven K-L settings on nine instances and two seeds.
- `results/csv/focused_mechanism.csv`: 126 focused runs and their run-level mechanism counters.
- `results/csv/qa_summary.csv`: independently reconciled manuscript aggregates.
- `results/csv/reported_statistics.csv`: paired W, Holm-adjusted p-value, and rank-biserial effect size records.
- `audit/mechanism_cases.json`: bounded success/failure case records retained during the focused experiment.
- `audit/selected_cases.json`: the two route-level cases reported in the manuscript.
- `figure_source/*.csv`: source values used for the benchmark, validation, K-marginal, and convergence figures.
- `benchmark_manifest.csv`: filenames, BKS values, byte counts, and SHA-256 hashes for the externally obtained Dang-82 files.

## Core variables

| Variable | Meaning |
|---|---|
| `Algorithm` | Reported algorithm or ablation variant. |
| `Instance` | Benchmark instance stem. |
| `Seed` | Pseudorandom seed (0-3 for the 82-instance studies; 0-1 for focused studies). |
| `Reward` | Final collected TOP reward; larger is better. |
| `BKS` | Published best-known-solution reward used as the reference. |
| `BKS_Gap_Percent_Source` | `100 * (BKS - Reward) / BKS`. |
| `Runtime_Seconds` | Wall-clock runtime per seed in seconds under the timing boundary described in the manuscript. |
| `Completed_Iterations` | Completed algorithm-specific search cycles. |
| `Budget_Seconds` | Instance-specific hard-wall reference budget. |
| `Within_Budget` | Whether measured runtime stayed within the assigned reference budget. |
| `Feasible` | Result of the common post-run feasibility check. |
| `Source_Record` | Provenance key linking the normalized row to its experiment block. |

The `Focused_Mechanism` table additionally reports repair calls, blocked and attempted customers, successful commits by ejection depth, bounded 2-opt participation, net reward gains, new-best co-occurrences, and module runtime.

## Provenance and processing

The JSON package was normalized from completed experiment manifests and result files without changing objective values, runtimes, iteration counts, or feasibility states. CSV tables are deterministic exports of that JSON. The Excel workbook contains the same records plus calculated gap columns and QA checks. Run-level gaps are computed directly from reward and BKS. Manuscript best-seed gaps first retain the best seed within each instance; mean-seed gaps average the four seeds within each instance; both are then averaged over instances.

Validate the complete package with:

```bash
python scripts/analysis/export_result_tables.py
python scripts/analysis/validate_reported_results.py
```

## Access and licence

Study-generated result data are available under CC BY 4.0; see `DATA_LICENSE.md`. The benchmark instances are reused public data and are not relicensed or redistributed. See `benchmarks/README.md` and `THIRD_PARTY.md`.

## Citation

Please cite the associated manuscript and the software repository. A persistent dataset DOI can be added by archiving a tagged GitHub release with Zenodo before manuscript submission.
