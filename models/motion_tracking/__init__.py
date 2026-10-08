"""Experimental motion association models."""
import os
import sys

# Match gt_process: avoid loading a second Intel OpenMP runtime on Windows.
if sys.platform == 'win32':
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
