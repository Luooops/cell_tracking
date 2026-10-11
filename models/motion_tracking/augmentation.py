"""Feature-level rigid transforms and conservative simulated missed detections."""
import math
import torch


def drop_detections(coords, features, parents, keep, max_gap, distance):
    """Relink through removed verified predecessors, never invent null labels."""
    selected = torch.nonzero(keep, as_tuple=True)[0]
    local = torch.full_like(parents, -1)
    local[selected] = torch.arange(len(selected), device=parents.device)
    remapped = parents[selected].clone()
    for j, original in enumerate(selected.tolist()):
        prior = int(parents[original])
        removed = False
        visited = set()
        while prior >= 0 and not bool(keep[prior]):
            if prior in visited:
                raise ValueError('cyclic predecessor labels')
            visited.add(prior)
            removed = True
            prior = int(parents[prior])
        if prior >= 0:
            dt = float(coords[original, 0] - coords[prior, 0])
            valid = 1 <= dt <= max_gap and float((coords[original, 1:] - coords[prior, 1:]).norm()) <= distance * dt
            remapped[j] = local[prior] if valid else -2
        else:
            remapped[j] = -2 if removed else prior
    if not (remapped != -2).any():
        return coords, features, parents
    return coords[selected], features[selected], remapped


def augment(coords, features, parents, max_gap, distance, detection_dropout=0.1, geometric=True):
    if detection_dropout:
        keep = torch.rand(len(coords), device=coords.device) >= detection_dropout
        # Unknown targets and positive sources may have unknown histories. Keep
        # all -2 tokens so incomplete labels cannot create spurious supervision.
        keep |= parents == -2
        coords, features, parents = drop_detections(coords, features, parents, keep, max_gap, distance)
    if geometric:
        coords = coords.clone()
        theta = float(torch.rand(())) * 2 * math.pi
        rotation = coords.new_tensor([[math.cos(theta), -math.sin(theta)],
                                      [math.sin(theta), math.cos(theta)]])
        center = coords[:, 1:].mean(0)
        coords[:, 1:] = (coords[:, 1:] - center) @ rotation.T + center
        if bool(torch.rand(()) < 0.5):
            coords[:, 1] = 2 * center[0] - coords[:, 1]
    return coords, features, parents
