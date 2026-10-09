from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class dishUseResult:
    """Occupancy maps and per-fish dish-use metrics.

    Fish order matches the input's fish axis. Below, F is the number of fish
    and N is ``grid_side_length`` (bins per axis). Map rows index y and columns index
    x; for example, ``occupancy_frame_counts[fish, y_bin, x_bin]``.

    Attributes
    ----------
    bin_edges : ndarray of float, shape (N + 1,)
        Shared x/y bin boundaries in millimeters, spanning the dish diameter.
    grid_x : ndarray of float, shape (N, N)
        x coordinate of each bin center, in millimeters.
    grid_y : ndarray of float, shape (N, N)
        y coordinate of each bin center, in millimeters.
    dish_mask : ndarray of bool, shape (N, N)
        True where a bin center lies inside or on the circular dish boundary.
    occupancy_frame_counts : ndarray of float, shape (F, N, N)
        Counts of retained sampled positions. Eligible unvisited bins contain
        zero; excluded bins contain NaN. Floating storage accommodates NaN.
    probability : ndarray of float, shape (F, N, N)
        Dimensionless occupancy counts divided by each fish's retained total.
        Eligible probabilities sum to one when occupancy is scorable;
        excluded bins are NaN. With no retained positions, the whole map is NaN.
    evenness : ndarray of float, shape (F,)
        Dimensionless normalized Shannon entropy: zero for one visited bin,
        one for equal occupancy across every eligible bin, NaN if unscorable.
    bias_index : ndarray of float, shape (F,)
        ``1 - evenness``; dimensionless and NaN if evenness is undefined.
    dish_bin_count : int
        Number of eligible bins, shared by every fish, including unvisited bins.
    visited_bin_count : ndarray of int, shape (F,)
        Number of eligible bins with at least one retained sampled position.
    coverage_fraction : ndarray of float, shape (F,)
        ``visited_bin_count / dish_bin_count``; dimensionless. Zero when no
        bins were visited, provided the grid has eligible bins.
    total_binned_frame_count : ndarray of int, shape (F,)
        Total positions retained in eligible bins; zero for empty occupancy.
    valid_in_dish_samples : ndarray of int, shape (F,)
        Finite x/y positions inside or on the circular boundary, before bin
        masking. This can exceed ``total_binned_frame_count`` near the edge.
    """

    bin_edges: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    dish_mask: np.ndarray
    occupancy_frame_counts: np.ndarray
    probability: np.ndarray
    evenness: np.ndarray
    bias_index: np.ndarray
    dish_bin_count: int
    visited_bin_count: np.ndarray
    coverage_fraction: np.ndarray
    total_binned_frame_count: np.ndarray
    valid_in_dish_samples: np.ndarray


def analyze_dish_use(
    xy_mm,
    dish_radius_mm,
    grid_side_length,
):
    """Calculate sampled-position occupancy and evenness on a common dish grid.

    Parameters
    ----------
    xy_mm : array_like, shape (samples, fish, 2)
        Sampled positions in millimeters, with the final axis ordered x, y.
        Each fish's coordinates must already be centered on its dish at (0, 0).
        Fish order is preserved in all returned per-fish arrays. This helper
        performs no pixel conversion, centering, rotation, or axis inversion.
        Positions with nonfinite x or y, or outside the dish, are excluded.
    dish_radius_mm : float
        Positive, finite expected dish radius in millimeters, shared by all fish.
    grid_side_length : int
        Number of equal-width cells per axis, at least two; booleans are rejected.
        This is a length in cells, not a physical length in millimeters.
        The square grid has ``grid_side_length ** 2`` bins before circular masking.

    Returns
    -------
    dishUseResult
        Shared grid geometry, occupancy/probability maps in (fish, y_bin, x_bin)
        order, and per-fish metrics. See the result class for every field's
        shape, units, and missing-data behavior.

    Raises
    ------
    ValueError
        If the coordinate shape is not (samples, fish, 2), the radius is
        nonpositive or nonfinite, or grid_side_length is not an integer of at least two.

    Notes
    -----
    Each retained position contributes one count. Counts refer to the supplied
    samples; elapsed time and frames omitted by sampling are not estimated.
    Eligible bins have centers inside or on the expected circular boundary.
    A position inside the circle may fall in an excluded edge bin, so
    ``valid_in_dish_samples`` can exceed ``total_binned_frame_count``.

    For each fish, p is occupancy divided by the retained count and K is the
    common ``dish_bin_count``. Evenness is ``-sum(p * log(p)) / log(K)``, using
    natural logarithms and positive probabilities only. Unvisited eligible bins
    have p = 0 and remain in K. Coverage is the fraction of eligible bins visited;
    bias_index is exactly 1 - evenness.

    Eligible unvisited bins have zero occupancy; excluded bins have NaN.
    With no retained positions, visited/total counts are zero, coverage is zero
    when eligible bins exist, and probability, evenness, and bias_index are NaN.
    Probability and evenness also remain undefined if K is at most one.

    Examples
    --------
    Two fish sampled four times: fish 0 stays in one bin, while fish 1 visits
    each of the four eligible bins once.

    >>> import numpy as np
    >>> xy_mm = np.full((4, 2, 2), -0.5)
    >>> xy_mm[:, 1, :] = [[-0.5, -0.5], [0.5, -0.5],
    ...                   [-0.5, 0.5], [0.5, 0.5]]
    >>> result = analyze_dish_use(
    ...     xy_mm, dish_radius_mm=1.0, grid_side_length=2)
    >>> np.testing.assert_allclose(result.evenness, [0.0, 1.0])
    >>> np.testing.assert_allclose(result.coverage_fraction, [0.25, 1.0])
    >>> result.total_binned_frame_count.tolist()
    [4, 4]
    """
    xy_mm = np.asarray(xy_mm, dtype=float)
    if xy_mm.ndim != 3 or xy_mm.shape[2] != 2:
        raise ValueError("xy_mm must have shape (samples, fish, 2).")

    dish_radius_mm = float(dish_radius_mm)
    if not np.isfinite(dish_radius_mm) or dish_radius_mm <= 0:
        raise ValueError("dish_radius_mm must be positive and finite.")
    if isinstance(grid_side_length, (bool, np.bool_)) or not isinstance(grid_side_length, Integral):
        raise ValueError("grid_side_length must be an integer of at least 2.")
    grid_side_length = int(grid_side_length)
    if grid_side_length < 2:
        raise ValueError("grid_side_length must be an integer of at least 2.")

    bin_edges = np.linspace(-dish_radius_mm, dish_radius_mm, grid_side_length + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(centers, centers)
    dish_mask = grid_x**2 + grid_y**2 <= dish_radius_mm**2
    dish_bin_count = int(dish_mask.sum())

    n_fish = xy_mm.shape[1]
    occupancy_frame_counts = np.full((n_fish, grid_side_length, grid_side_length), np.nan, dtype=float)
    probability = np.full_like(occupancy_frame_counts, np.nan)
    evenness = np.full(n_fish, np.nan, dtype=float)
    visited_bin_count = np.zeros(n_fish, dtype=int)
    coverage_fraction = np.full(n_fish, np.nan, dtype=float)
    total_binned_frame_count = np.zeros(n_fish, dtype=int)
    valid_in_dish_samples = np.zeros(n_fish, dtype=int)

    for fish_index in range(n_fish):
        xy = xy_mm[:, fish_index, :]
        finite = np.isfinite(xy).all(axis=1)
        radial_distance = np.hypot(xy[:, 0], xy[:, 1])
        valid_in_dish = finite & (radial_distance <= dish_radius_mm)
        valid_in_dish_samples[fish_index] = int(valid_in_dish.sum())

        histogram, _, _ = np.histogram2d(
            xy[valid_in_dish, 0],
            xy[valid_in_dish, 1],
            bins=(bin_edges, bin_edges),
        )
        occupancy = histogram.T
        occupancy[~dish_mask] = np.nan
        occupancy_frame_counts[fish_index] = occupancy

        positive = dish_mask & np.isfinite(occupancy) & (occupancy > 0)
        visited_bin_count[fish_index] = int(positive.sum())
        total = int(occupancy[positive].sum()) if positive.any() else 0
        total_binned_frame_count[fish_index] = total
        if dish_bin_count:
            coverage_fraction[fish_index] = (
                visited_bin_count[fish_index] / dish_bin_count
            )

        if dish_bin_count <= 1 or total <= 0:
            continue

        fish_probability = np.zeros((grid_side_length, grid_side_length), dtype=float)
        fish_probability[~dish_mask] = np.nan
        fish_probability[dish_mask] = occupancy[dish_mask] / total
        probability[fish_index] = fish_probability

        positive_probabilities = fish_probability[positive]
        entropy = -np.sum(positive_probabilities * np.log(positive_probabilities))
        evenness[fish_index] = float(entropy / np.log(dish_bin_count))

    bias_index = np.where(np.isfinite(evenness), 1 - evenness, np.nan)
    return dishUseResult(
        bin_edges=bin_edges,
        grid_x=grid_x,
        grid_y=grid_y,
        dish_mask=dish_mask,
        occupancy_frame_counts=occupancy_frame_counts,
        probability=probability,
        evenness=evenness,
        bias_index=bias_index,
        dish_bin_count=dish_bin_count,
        visited_bin_count=visited_bin_count,
        coverage_fraction=coverage_fraction,
        total_binned_frame_count=total_binned_frame_count,
        valid_in_dish_samples=valid_in_dish_samples,
    )
