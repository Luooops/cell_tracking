"""Regenerate every paper figure and table (inputs: development/final_results, data/).
    python paper/figures/make_all.py [--refresh] [--out DIR]
--refresh recomputes the cached per-gap link counts; --out is where the figures and tables.tex go (default
paper/figures/out). The LaTeX sources live in their own repository, D:\\Papers\\GNNCellTracking-Paper, which reads
them from figures/out: --out D:/Papers/GNNCellTracking-Paper/figures/out."""
import argparse
import shutil
import sys
from pathlib import Path

import data
import style
import fig_confidence
import fig_division
import fig_gaps
import fig_main
import fig_qualitative
import fig_suppl
import tables

ROOT = Path(__file__).resolve().parents[2]
# Figures whose master is an SVG edited in Inkscape: copied, never regenerated here. The main figure's master
# development/overview_design/challenges/fig.svg was edited by hand (fig.pdf/png exported from it with Inkscape);
# the framework figure is drawn by fig_framework.py into development/overview_design/framework/.
MASTERS = {'fig0_main': ROOT / 'development' / 'overview_design' / 'challenges',
           'fig_framework': ROOT / 'development' / 'overview_design' / 'framework'}


def copy_masters():
    for name, folder in MASTERS.items():
        for ext in ('pdf', 'png'):
            shutil.copyfile(folder / f'fig.{ext}', style.OUT / f'{name}.{ext}')
        print('copied', style.OUT / f'{name}.pdf')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--out', type=Path, default=style.OUT)
    args = parser.parse_args()
    style.OUT = args.out
    data.gap_counts(refresh=args.refresh)
    style.OUT.mkdir(parents=True, exist_ok=True)
    copy_masters()
    fig_main.main()
    fig_gaps.main()
    fig_confidence.main()
    sys.argv = sys.argv[:1]                     # automatic region choice for figure 4
    fig_qualitative.main()
    fig_suppl.main()
    fig_division.main(refresh=args.refresh)
    tables.main()


if __name__ == '__main__':
    main()
