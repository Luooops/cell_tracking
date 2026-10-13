"""Mirror the console output of one run into <output_root>/logs/<model>/<version>_<stage>_<dataset>.log.

The console keeps printing as before; stdout and stderr (including an uncaught traceback) are also appended to the
log file between a header (time, command, git commit) and a footer (status, duration). Re-running the same name
appends a new section instead of overwriting. Spawned worker processes join the same file with tee_process().
Output written directly by C extensions is not captured."""
import datetime
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log_path(output_root, model, name):
    return Path(output_root) / 'logs' / model / f'{name}.log'


def log_root(target, output_root):
    """The configured output root when target lies inside it, else target's parent (keeps scratch runs out)."""
    target, output_root = Path(target).resolve(), Path(output_root).resolve()
    return output_root if output_root in target.parents else target.parent


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _git_commit():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, timeout=10)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return (result.stdout.strip() or 'unknown') + (' (uncommitted changes)' if dirty.stdout.strip() else '')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def tee_process(path):
    """In a child process: also append this process's stdout/stderr to the parent's log file."""
    if path:
        log = open(path, 'a', encoding='utf-8', buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)


def run_logged(path, func, *args, argv=None):
    """Call func(*args) while teeing stdout/stderr into path; returns func's result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ' '.join([Path(sys.argv[0]).name] + list(sys.argv[1:] if argv is None else argv))
    start = time.time()
    with path.open('a', encoding='utf-8', buffering=1) as log:     # line-buffered: workers append too
        log.write(f"==== {datetime.datetime.now().isoformat(timespec='seconds')} ====\n"
                  f'command: {command}\npython: {platform.python_version()} ({sys.executable})\n'
                  f'git: {_git_commit()}\n\n')
        log.flush()
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)
        status = 'failed'
        try:
            result = func(*args)
            status = 'complete' if result in (0, None) else f'exit {result}'
            return result
        except SystemExit as exc:
            status = 'complete' if exc.code in (0, None) else f'exit {exc.code}'
            raise
        except BaseException:
            log.write(traceback.format_exc())
            raise
        finally:
            sys.stdout, sys.stderr = out, err
            log.write(f'\n==== {status} after {time.time() - start:.1f} s ====\n\n')
