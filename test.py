"""Unified inference entry; model-specific behavior is implemented in adapters."""
from utils.config import options


def main(argv=None):
    from utils.runners import run_test
    return run_test(options('test', argv))


if __name__ == '__main__':
    raise SystemExit(main())
