#!/usr/bin/env bash
# QUELL v4.0 — regenerate every derived number, Tables II-XI and Figures 4-6 from the committed result files, then
# check the numbers printed in the manuscript's tables against those files. CPU only, no data download, a few seconds.
# Usage: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/21_derived_stats.py   > results/v4/derived_stats.json          # seeds, CIs, Glass's Δ, summaries
python3 scripts/20_jetson_summary.py  > results/v4/jetson/jetson_summary.json  # on-device measurements
python3 figures/make_fig4.py                                                             # cost-accuracy trade-off
python3 figures/make_fig5.py                                                             # unknown-attack generalization
python3 figures/make_fig6.py                                                             # random-noise sensitivity
python3 scripts/22_verify_numbers.py                                              # manuscript tables == result files
python3 scripts/23_make_tables.py > /dev/null                                     # Tables II-XI -> results/v4/tables/*.csv
git diff --stat -- results figures 2>/dev/null || true                                   # empty = bit-identical regeneration
