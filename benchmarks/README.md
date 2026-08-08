# Benchmark data

The study uses 82 Dang instances and 157 Chao instances from Sets 4-7. Third-party instance files are not redistributed in this repository.

1. Obtain the Dang benchmark from <https://www.hds.utc.fr/~moukrim/dokuwiki/en/top> and place the 82 files in `benchmarks/Dang et al., (2013)/`.
2. Obtain the canonical Chao TOP collection from its original/public distribution and place Sets 4-7 in:
   - `benchmarks/Chao et al., (1996)/Set 4/`
   - `benchmarks/Chao et al., (1996)/Set 5/`
   - `benchmarks/Chao et al., (1996)/Set 6/`
   - `benchmarks/Chao et al., (1996)/Set 7/`
3. Verify all files against the published manifest:

```bash
python scripts/analysis/build_benchmark_manifest.py benchmarks \
  --output outputs/benchmark_manifest_check.csv
```

Compare the result with `data/benchmark_manifest.csv`. The expected collection contains 239 files: 82 Dang and 157 Chao instances with published BKS values. The manifest permits exact verification without redistributing either benchmark collection.
