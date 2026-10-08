"""Lazy model selection; importing this package does not load GPU libraries."""
from importlib import import_module

MODEL_MODULES = {
    'gnn_tracking': 'models.gnn_tracking.track',
    'gnn_division': 'models.gnn_tracking.track_division',
    'trackastra': 'models.trackastra.run_trackastra',
    'motion_tracking': 'models.motion_tracking.model',
    'classical_tracking': 'models.classical_tracking.tracking_v0',
    'classical_legacy': 'models.classical_tracking.legacy',
    'mitosis': 'models.classical_tracking.mitosis',
}


def get_model(name):
    """Return the existing implementation module, without wrapping its algorithms."""
    if name not in MODEL_MODULES:
        raise ValueError(f'Unknown model {name!r}; choose from {", ".join(MODEL_MODULES)}')
    return import_module(MODEL_MODULES[name])
