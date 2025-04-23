import numpy as np
from typing import List

DEPTH_PRUNE_STRATEGY_LIST = ["firstlast", "first", "last"]

def get_kept_layers_prune(strategy: str, n_layers: int, ratio: float) -> List[int]:
    n_layers_kept = int((1 - ratio) * n_layers)
    if strategy == "firstlast":
        # first, last N - 1 layers
        idx = [0] + list(range(n_layers - n_layers_kept + 1, n_layers))
    elif strategy == "first":
        idx = list(range(n_layers_kept))
    elif strategy == "last":
        idx = list(range(n_layers - n_layers_kept, n_layers))
    else:
        raise ValueError(f"invalid depth pruning strategy {strategy}")
    assert len(idx) == n_layers_kept
    return idx

def get_kept_layers(n_teacher_layers: int, n_student_layers: int, strategy: str="equal_span") -> List[int]:
    """
    Select n_student_layers from n_teacher_layers with equal spacing.
    
    Args:
        n_teacher_layers: Number of layers in the teacher model
        n_student_layers: Number of layers in the student model
        strategy: Strategy for selecting layers (currently only "equal_span" is supported)
    
    Returns:
        List of layer indices to keep from the teacher model
    """
    if strategy != "equal_span":
        raise ValueError(f"Only 'equal_span' strategy is supported for this function")
    
    if n_student_layers >= n_teacher_layers:
        return list(range(n_teacher_layers))
    
    # Use np.arange to get evenly spaced indices
    indices = np.linspace(0, n_teacher_layers - 1, n_student_layers)
    # Round to nearest integer and convert to list
    return [int(round(x)) for x in indices]
