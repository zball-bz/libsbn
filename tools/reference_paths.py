"""Optional experiment/reference locations; never required by library builds."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def experiments_root(required=False):
    root = Path(os.environ.get('SBN_EXPERIMENTS_ROOT', ROOT/'experiments')).expanduser().resolve()
    if required and not (root/'config/benchmarks.json').is_file():
        raise SystemExit('Experiments are optional. Clone zball-bz/libsbn_experiments into experiments/ or set SBN_EXPERIMENTS_ROOT.')
    return root

def reference_root(required=False):
    root = Path(os.environ.get('SBN_REFERENCE_ROOT', experiments_root()/'reference/legacy')).expanduser().resolve()
    if required and not root.is_dir():
        raise SystemExit('Missing frozen reference pack; clone libsbn_experiments or set SBN_REFERENCE_ROOT.')
    return root

def reference_path(name):
    text = str(name)
    if text.startswith('../'):
        return str(reference_root(True)/text[3:])
    if text.startswith('bench/'):
        return str(experiments_root(True)/text)
    return text

def reference_command(command):
    result = []
    for value in command:
        text = str(value)
        if text.startswith('-I../'):
            text = '-I'+reference_path(text[2:])
        elif text.startswith(('../','bench/')):
            text = reference_path(text)
        result.append(text)
    return result
