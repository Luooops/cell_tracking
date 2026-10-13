"""Unified inference entry; model-specific behavior is implemented in adapters."""
from utils.config import options
from utils.run_log import run_logged


def main(argv=None):
    from utils.runners import run_test
    a = options('test', argv)
    return run_logged(a.log_path, run_test, a, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
