# Third-party resources

## Dang team orienteering benchmark

The 82 benchmark instance files used in the manuscript are not redistributed. They originate from:

- D.-C. Dang, R. N. Guibadj, and A. Moukrim, "An effective PSO-inspired algorithm for the team orienteering problem," *European Journal of Operational Research* 229(2), 332-344 (2013). https://doi.org/10.1016/j.ejor.2013.02.049
- Original benchmark page: https://www.hds.utc.fr/~moukrim/dokuwiki/en/top

`data/benchmark_manifest.csv` provides the expected filenames, BKS values, byte counts, and SHA-256 checksums for the exact 82 files used.

## PyVRP

The repository contains only a TOP adapter. PyVRP itself is installed from the externally maintained `pyvrp` package and is not vendored here. The reported runs used PyVRP 0.13.4. Project: https://github.com/PyVRP/PyVRP

## Kim augmented LNS

`src/top_alns/kim_alns` is an implementation-level reproduction written for this study from the algorithms described in:

- B.-I. Kim, H. Li, and A. L. Johnson, "An augmented large neighborhood search method for solving the team orienteering problem," *Expert Systems with Applications* 40(8), 3065-3072 (2013). https://doi.org/10.1016/j.eswa.2012.12.022

No third-party source code from the article is redistributed.

## GRASP, ILS, and VNS

The reported GRASP, iterated local search, and variable neighbourhood search comparators are the in-house implementations under `src/top_alns/comparison_baselines` and are covered by this repository's MIT licence.
