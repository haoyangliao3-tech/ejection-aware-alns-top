# Benchmark data

The manuscript uses the 82 larger TOP instances introduced by Dang, Guibadj, and Moukrim (2013). The instance files are not redistributed in this repository.

1. Obtain the benchmark from https://www.hds.utc.fr/~moukrim/dokuwiki/en/top.
2. Place the 82 `.txt` files in `benchmarks/Dang et al., (2013)/`.
3. Verify that the files match the study:

```bash
python scripts/analysis/build_benchmark_manifest.py \
  "benchmarks/Dang et al., (2013)" \
  --output outputs/benchmark_manifest_check.csv
```

Compare the generated manifest with `data/benchmark_manifest.csv`. The expected set has exactly 82 files.
