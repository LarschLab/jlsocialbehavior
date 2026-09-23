from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ArenaEvennessResult:
    """Binned arena occupancy and its normalized Shannon evenness."""

    bin_edges: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    arena_mask: np.ndarray
    occupancy_seconds: np.ndarray
    probability: np.ndarray
    evenness: np.ndarray
    bias_index: np.ndarray
    arena_bin_count: int
    visited_bin_count: np.ndarray
    coverage_fraction: np.ndarray
    total_in_arena_seconds: np.ndarray
    valid_in_arena_samples: np.ndarray


def calculate_arena_use_evenness(
    xy_mm,
    arena_radius_mm,
    bin_count,
    sample_interval_seconds,
):
    """Calculate per-fish occupancy and evenness over the expected arena disk.

    The square grid spans the expected arena diameter. Bin centers inside the
    circular arena define the common set of eligible bins for every fish.
    Unvisited eligible bins remain in the Shannon-normalization denominator.
    """
    xy_mm = np.asarray(xy_mm, dtype=float)
    if xy_mm.ndim != 3 or xy_mm.shape[2] != 2:
        raise ValueError("xy_mm must have shape (samples, fish, 2).")

    arena_radius_mm = float(arena_radius_mm)
    if not np.isfinite(arena_radius_mm) or arena_radius_mm <= 0:
        raise ValueError("arena_radius_mm must be positive and finite.")
    if isinstance(bin_count, (bool, np.bool_)) or not isinstance(bin_count, Integral):
        raise ValueError("bin_count must be an integer of at least 2.")
    bin_count = int(bin_count)
    if bin_count < 2:
        raise ValueError("bin_count must be an integer of at least 2.")

    sample_interval_seconds = float(sample_interval_seconds)
    if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive and finite.")

    bin_edges = np.linspace(-arena_radius_mm, arena_radius_mm, bin_count + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(centers, centers)
    arena_mask = grid_x**2 + grid_y**2 <= arena_radius_mm**2
    arena_bin_count = int(arena_mask.sum())

    n_fish = xy_mm.shape[1]
    occupancy_seconds = np.full((n_fish, bin_count, bin_count), np.nan, dtype=float)
    probability = np.full_like(occupancy_seconds, np.nan)
    evenness = np.full(n_fish, np.nan, dtype=float)
    visited_bin_count = np.zeros(n_fish, dtype=int)
    coverage_fraction = np.full(n_fish, np.nan, dtype=float)
    total_in_arena_seconds = np.zeros(n_fish, dtype=float)
    valid_in_arena_samples = np.zeros(n_fish, dtype=int)

    for fish_index in range(n_fish):
        xy = xy_mm[:, fish_index, :]
        finite = np.isfinite(xy).all(axis=1)
        radial_distance = np.hypot(xy[:, 0], xy[:, 1])
        valid_in_arena = finite & (radial_distance <= arena_radius_mm)
        valid_in_arena_samples[fish_index] = int(valid_in_arena.sum())

        histogram, _, _ = np.histogram2d(
            xy[valid_in_arena, 0],
            xy[valid_in_arena, 1],
            bins=(bin_edges, bin_edges),
        )
        occupancy = histogram.T * sample_interval_seconds
        occupancy[~arena_mask] = np.nan
        occupancy_seconds[fish_index] = occupancy

        positive = arena_mask & np.isfinite(occupancy) & (occupancy > 0)
        visited_bin_count[fish_index] = int(positive.sum())
        total = float(occupancy[positive].sum()) if positive.any() else 0.0
        total_in_arena_seconds[fish_index] = total
        if arena_bin_count:
            coverage_fraction[fish_index] = (
                visited_bin_count[fish_index] / arena_bin_count
            )

        if arena_bin_count <= 1 or total <= 0:
            continue

        fish_probability = np.zeros((bin_count, bin_count), dtype=float)
        fish_probability[~arena_mask] = np.nan
        fish_probability[arena_mask] = occupancy[arena_mask] / total
        probability[fish_index] = fish_probability

        positive_probabilities = fish_probability[positive]
        entropy = -np.sum(positive_probabilities * np.log(positive_probabilities))
        evenness[fish_index] = float(entropy / np.log(arena_bin_count))

    bias_index = np.where(np.isfinite(evenness), 1 - evenness, np.nan)
    return ArenaEvennessResult(
        bin_edges=bin_edges,
        grid_x=grid_x,
        grid_y=grid_y,
        arena_mask=arena_mask,
        occupancy_seconds=occupancy_seconds,
        probability=probability,
        evenness=evenness,
        bias_index=bias_index,
        arena_bin_count=arena_bin_count,
        visited_bin_count=visited_bin_count,
        coverage_fraction=coverage_fraction,
        total_in_arena_seconds=total_in_arena_seconds,
        valid_in_arena_samples=valid_in_arena_samples,
    )
