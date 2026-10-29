"""Regenerate every paper figure and table (inputs: development/final_results, data/).
    python paper/figures/make_all.py [--refresh] [--out DIR]
--refresh recomputes the cached per-gap link counts; --out is where the figures and tables.tex go (default
paper/figures/out). The LaTeX sources live in their own repository, D:\\Papers\\GNNCellTracking-Paper, which reads
them from figures/out: --out D:/Papers/GNNCellTracking-Paper/figures/out."""
import argparse
import sys
from pathlib import Path

import data
import style
import fig_confidence
import fig_gaps
import fig_main
import fig_overview
import fig_qualitative
import fig_suppl
import tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--out', type=Path, default=style.OUT)
    args = parser.parse_args()
    style.OUT = args.out
    data.gap_counts(refresh=args.refresh)
    fig_overview.main()
    fig_main.main()
    fig_gaps.main()
    fig_confidence.main()
    sys.argv = sys.argv[:1]                     # automatic region choice for figure 4
    fig_qualitative.main()
    fig_suppl.main()
    tables.main()


if __name__ == '__main__':
    main()
