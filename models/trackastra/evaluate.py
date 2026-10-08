"""Evaluate Trackastra outputs with the project's shared tracking metrics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.legacy_evaluate import main as evaluate_main


def main(argv=None):
    return evaluate_main([
        "--predictions-root", str(ROOT / "outputs" / "trackastra" / "legacy"),
        "--output-root", str(ROOT / "outputs" / "trackastra" / "legacy_evaluation"),
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    sys.exit(main())
