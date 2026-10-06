"""Evaluate Trackastra outputs with the project's shared tracking metrics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from track_eval.evaluate import main as evaluate_main


def main(argv=None):
    return evaluate_main([
        "--predictions-root", str(ROOT / "trackastra_tracking" / "outputs"),
        "--output-root", str(ROOT / "trackastra_tracking" / "evaluation_outputs"),
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    sys.exit(main())
