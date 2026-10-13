from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


LOOM_CACHE_SCHEMA_VERSION = 7
POOLED_GENOTYPE = "__pooled__"
SUPPORTED_LOOM_EPISODE_PATTERN = re.compile(r"^CL(?:full|semi)(\d{3})([LR])")
UNSIDED_LOOM_EPISODE_PATTERN = re.compile(r"^CLsemi(\d{3})$")


class _NoLoomStimulusEpisodesError(ValueError):
    """Signal that an otherwise readable experiment has no supported loom episodes."""


STIMULUS_COLUMNS = [
    "episode",
    "stim_x",
    "stim_y",
    "stim_size",
    "background_contrast",
    "foreground_contrast",
]


def animal_column_names(n_animals: int = 35) -> list[str]:
    columns: list[str] = []
    for animal in range(1, n_animals + 1):
        columns.extend(
            [
                f"animal_{animal:02d}_x",
                f"animal_{animal:02d}_y",
                f"animal_{animal:02d}_orientation",
            ]
        )
    columns.extend(["stim_x_embedded", "stim_y_embedded", "stim_size_embedded", "episode_embedded"])
    return columns


def load_stimulus_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=STIMULUS_COLUMNS)
    df["frame"] = np.arange(len(df))
    df["block"] = df["episode"].ne(df["episode"].shift()).cumsum() - 1
    df["frame_in_block"] = df.groupby("block").cumcount()
    return df


def load_animal_file(path: str | Path, n_animals: int = 35) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=animal_column_names(n_animals),
        na_values=["nan"],
    )


def embedded_stimulus_from_animal_file(animal_df: pd.DataFrame) -> pd.DataFrame:
    stimulus_df = animal_df[
        ["episode_embedded", "stim_x_embedded", "stim_y_embedded", "stim_size_embedded"]
    ].rename(
        columns={
            "episode_embedded": "episode",
            "stim_x_embedded": "stim_x",
            "stim_y_embedded": "stim_y",
            "stim_size_embedded": "stim_size",
        }
    )
    stimulus_df = stimulus_df.copy()
    stimulus_df["frame"] = np.arange(len(stimulus_df))
    stimulus_df["block"] = stimulus_df["episode"].ne(stimulus_df["episode"].shift()).cumsum() - 1
    stimulus_df["frame_in_block"] = stimulus_df.groupby("block").cumcount()
    return stimulus_df


def stimulus_blocks(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    blocks = (
        stimulus_df.groupby("block", sort=False)
        .agg(
            episode=("episode", "first"),
            start_frame=("frame", "first"),
            end_frame=("frame", "last"),
            n_frames=("frame", "size"),
            stim_x_min=("stim_x", "min"),
            stim_x_max=("stim_x", "max"),
            stim_y_min=("stim_y", "min"),
            stim_y_max=("stim_y", "max"),
            stim_size_min=("stim_size", "min"),
            stim_size_max=("stim_size", "max"),
        )
        .reset_index()
    )

    blocks["category"] = np.select(
        [blocks["episode"].str.startswith("CL"), blocks["episode"].str.startswith("gr")],
        ["closed-loop loom", "grating"],
        default="moving trajectory",
    )
    return blocks


def loom_trials(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    blocks = stimulus_blocks(stimulus_df)
    cl_blocks = blocks.loc[blocks["category"].eq("closed-loop loom")].copy()

    trials = []
    for trial_index, block_row in enumerate(cl_blocks.itertuples(index=False)):
        segment = stimulus_df.loc[stimulus_df["block"].eq(block_row.block)]
        positive_size = segment.loc[segment["stim_size"].gt(0)]
        onset_frame = int(positive_size["frame"].iloc[0]) if len(positive_size) else int(block_row.start_frame)

        match = SUPPORTED_LOOM_EPISODE_PATTERN.match(block_row.episode)
        loom_max_size = int(match.group(1)) if match else int(round(block_row.stim_size_max))
        side = match.group(2) if match else ""

        trials.append(
            {
                "trial": trial_index,
                "block": int(block_row.block),
                "episode": block_row.episode,
                "side": side,
                "loom_max_size": loom_max_size,
                "block_start_frame": int(block_row.start_frame),
                "loom_onset_frame": onset_frame,
                "onset_in_block": onset_frame - int(block_row.start_frame),
                "block_end_frame": int(block_row.end_frame),
                "n_frames": int(block_row.n_frames),
            }
        )

    return pd.DataFrame(trials)


def unsided_loom_trials(
    stimulus_df: pd.DataFrame,
    onset_in_block: int = 150,
) -> pd.DataFrame:
    """Split unsided ``CLsemi###`` blocks into their individual loom pulses.

    The unsided tuning protocol stores several center looms inside one long
    episode block. Each transition from zero to positive stimulus size is one
    trial. ``block_start_frame`` is a virtual trial start chosen so the pulse
    onset retains the legacy fixed-onset frame used by response metrics.
    """
    if onset_in_block < 0:
        raise ValueError("onset_in_block must be nonnegative.")

    blocks = stimulus_blocks(stimulus_df)
    rows = []
    trial_index = 0
    for block_row in blocks.itertuples(index=False):
        match = UNSIDED_LOOM_EPISODE_PATTERN.fullmatch(str(block_row.episode))
        if match is None:
            continue

        segment = stimulus_df.loc[stimulus_df["block"].eq(block_row.block)].sort_values("frame")
        positive = segment["stim_size"].fillna(0).gt(0)
        pulse_onsets = segment.loc[positive & ~positive.shift(fill_value=False), "frame"].astype(int)
        for pulse_in_block, pulse_onset in enumerate(pulse_onsets, start=1):
            if pulse_in_block < len(pulse_onsets):
                pulse_end = int(pulse_onsets.iloc[pulse_in_block]) - 1
            else:
                pulse_end = int(block_row.end_frame)
            rows.append(
                {
                    "trial": trial_index,
                    "block": int(block_row.block),
                    "pulse_in_block": pulse_in_block,
                    "episode": str(block_row.episode),
                    "side": "",
                    "loom_max_size": int(match.group(1)),
                    "block_start_frame": int(pulse_onset) - onset_in_block,
                    "loom_onset_frame": int(pulse_onset),
                    "onset_in_block": onset_in_block,
                    "block_end_frame": pulse_end,
                    "n_frames": pulse_end - (int(pulse_onset) - onset_in_block) + 1,
                }
            )
            trial_index += 1

    return pd.DataFrame(rows)


def grating_trials(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    blocks = stimulus_blocks(stimulus_df)
    grating_blocks = blocks.loc[blocks["category"].eq("grating")].copy()

    trials = []
    for trial_index, block_row in enumerate(grating_blocks.itertuples(index=False)):
        match = re.match(r"gr(?:OL|at)(\d{3})(\d{3})(\d{3})([LR])", block_row.episode)
        if match:
            prefix_code = int(match.group(1))
            frequency_code = int(match.group(2))
            suffix_code = int(match.group(3))
            direction = match.group(4)
        else:
            prefix_code = np.nan
            frequency_code = np.nan
            suffix_code = np.nan
            direction = ""

        trials.append(
            {
                "trial": trial_index,
                "block": int(block_row.block),
                "episode": block_row.episode,
                "direction": direction,
                "frequency_code": frequency_code,
                "prefix_code": prefix_code,
                "suffix_code": suffix_code,
                "block_start_frame": int(block_row.start_frame),
                "block_end_frame": int(block_row.end_frame),
                "n_frames": int(block_row.n_frames),
            }
        )

    return pd.DataFrame(trials)


def animal_arrays(animal_df: pd.DataFrame, n_animals: int = 35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cols = [f"animal_{animal:02d}_x" for animal in range(1, n_animals + 1)]
    y_cols = [f"animal_{animal:02d}_y" for animal in range(1, n_animals + 1)]
    orientation_cols = [f"animal_{animal:02d}_orientation" for animal in range(1, n_animals + 1)]
    return (
        animal_df[x_cols].to_numpy(dtype=float),
        animal_df[y_cols].to_numpy(dtype=float),
        animal_df[orientation_cols].to_numpy(dtype=float),
    )


def extract_loom_snippets(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    pre_frames: int = 10,
    post_frames: int = 60,
    n_animals: int = 35,
    north_angle: float = np.pi / 2,
) -> pd.DataFrame:
    x, y, orientation = animal_arrays(animal_df, n_animals=n_animals)
    relative_frames = np.arange(-pre_frames, post_frames + 1)
    snippets = []

    for trial_row in trials_df.itertuples(index=False):
        onset = int(trial_row.loom_onset_frame)
        start = onset - pre_frames
        stop = onset + post_frames + 1
        if start < 0 or stop > len(animal_df):
            continue

        window_x = x[start:stop]
        window_y = y[start:stop]
        onset_x = x[onset]
        onset_y = y[onset]
        onset_orientation = orientation[onset]

        dx = window_x - onset_x
        # Animal positions are image coordinates: x increases right, y increases
        # downward. Convert y to a conventional plotting axis before rotating.
        dy = -(window_y - onset_y)

        heading_angle = -onset_orientation
        rotation = north_angle - heading_angle
        cos_rotation = np.cos(rotation)
        sin_rotation = np.sin(rotation)

        ego_x = dx * cos_rotation - dy * sin_rotation
        ego_y = dx * sin_rotation + dy * cos_rotation
        distance = np.hypot(ego_x, ego_y)

        for animal_index in range(n_animals):
            snippets.append(
                pd.DataFrame(
                    {
                        "trial": int(trial_row.trial),
                        "block": int(trial_row.block),
                        "episode": trial_row.episode,
                        "side": trial_row.side,
                        "loom_max_size": int(trial_row.loom_max_size),
                        "animal": animal_index + 1,
                        "relative_frame": relative_frames,
                        "global_frame": np.arange(start, stop),
                        "ego_x": ego_x[:, animal_index],
                        "ego_y": ego_y[:, animal_index],
                        "distance_from_onset": distance[:, animal_index],
                        "raw_x": window_x[:, animal_index],
                        "raw_y": window_y[:, animal_index],
                        "orientation_at_onset": onset_orientation[animal_index],
                    }
                )
            )

    return pd.concat(snippets, ignore_index=True)


def extract_legacy_loom_snippets(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    stimulus_df: pd.DataFrame,
    pre_frames: int = 50,
    post_frames: int = 99,
    onset_in_block: int = 150,
    n_animals: int = 35,
    units_per_mm: float = 4.0,
) -> pd.DataFrame:
    """Extract loom snippets using the legacy notebook's rotation convention.

    Legacy behavior:
    - Align every condition to a fixed frame within the 2700-frame CL block.
    - Subtract animal position at that fixed onset frame.
    - Rotate animal and stimulus positions by ``-orientation + pi/2``.
    - Do not flip image y and do not mirror L/R.
    """
    x, y, orientation = animal_arrays(animal_df, n_animals=n_animals)
    stim_x = stimulus_df["stim_x"].to_numpy(dtype=float)
    stim_y = stimulus_df["stim_y"].to_numpy(dtype=float)
    stim_size = stimulus_df["stim_size"].to_numpy(dtype=float)
    relative_frames = np.arange(-pre_frames, post_frames + 1)
    snippets = []

    for trial_row in trials_df.itertuples(index=False):
        onset = int(trial_row.block_start_frame) + onset_in_block
        start = onset - pre_frames
        stop = onset + post_frames + 1
        if start < 0 or stop > len(animal_df):
            continue

        window_x = x[start:stop]
        window_y = y[start:stop]
        onset_x = x[onset]
        onset_y = y[onset]
        onset_orientation = orientation[onset]

        animal_dx = window_x - onset_x
        animal_dy = window_y - onset_y
        stim_dx = stim_x[start:stop, None] - onset_x
        stim_dy = stim_y[start:stop, None] - onset_y

        animal_theta = np.arctan2(animal_dy, animal_dx)
        animal_radius = np.hypot(animal_dx, animal_dy)
        stim_theta = np.arctan2(stim_dy, stim_dx)
        stim_radius = np.hypot(stim_dx, stim_dy)

        rotation = -onset_orientation + np.pi / 2
        animal_theta_rot = animal_theta + rotation
        stim_theta_rot = stim_theta + rotation

        legacy_x = animal_radius * np.cos(animal_theta_rot)
        legacy_y = animal_radius * np.sin(animal_theta_rot)
        legacy_stim_x = stim_radius * np.cos(stim_theta_rot)
        legacy_stim_y = stim_radius * np.sin(stim_theta_rot)
        center_dist = np.hypot(legacy_x, legacy_y)

        step_dx = np.diff(window_x, axis=0, prepend=np.nan)
        step_dy = np.diff(window_y, axis=0, prepend=np.nan)
        step_dist = np.hypot(step_dx, step_dy)

        for animal_index in range(n_animals):
            snippets.append(
                pd.DataFrame(
                    {
                        "trial": int(trial_row.trial),
                        "block": int(trial_row.block),
                        "episode": trial_row.episode,
                        "side": trial_row.side,
                        "loom_max_size": int(trial_row.loom_max_size),
                        "animal": animal_index + 1,
                        "relative_frame": relative_frames,
                        "epFrame": onset_in_block + relative_frames,
                        "global_frame": np.arange(start, stop),
                        "legacy_x": legacy_x[:, animal_index],
                        "legacy_y": legacy_y[:, animal_index],
                        "legacy_stim_x": legacy_stim_x[:, animal_index],
                        "legacy_stim_y": legacy_stim_y[:, animal_index],
                        "center_dist": center_dist[:, animal_index],
                        "center_dist_mm": center_dist[:, animal_index] / units_per_mm,
                        "step_dist": step_dist[:, animal_index],
                        "raw_x": window_x[:, animal_index],
                        "raw_y": window_y[:, animal_index],
                        "orientation_at_onset": onset_orientation[animal_index],
                        "stim_size": stim_size[start:stop],
                    }
                )
            )

    return pd.concat(snippets, ignore_index=True)


def legacy_loom_response_metrics(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    baseline_frames: tuple[int, int] = (100, 140),
    response_frames: tuple[int, int] = (149, 180),
    n_animals: int = 35,
) -> pd.DataFrame:
    """Compute legacy baseline/response distance and binary escape metric."""
    x, y, _ = animal_arrays(animal_df, n_animals=n_animals)
    rows = []

    for trial_row in trials_df.itertuples(index=False):
        block_start = int(trial_row.block_start_frame)
        baseline_start = block_start + baseline_frames[0]
        baseline_end = block_start + baseline_frames[1]
        response_start = block_start + response_frames[0]
        response_end = block_start + response_frames[1]
        if baseline_start < 0 or response_end >= len(animal_df):
            continue

        baseline_dist = np.hypot(x[baseline_end] - x[baseline_start], y[baseline_end] - y[baseline_start])
        response_dist = np.hypot(x[response_end] - x[response_start], y[response_end] - y[response_start])
        ratio = np.divide(
            response_dist,
            baseline_dist,
            out=np.zeros_like(response_dist, dtype=float),
            where=np.isfinite(baseline_dist) & (baseline_dist != 0),
        )

        for animal_index in range(n_animals):
            rows.append(
                {
                    "trial": int(trial_row.trial),
                    "block": int(trial_row.block),
                    "episode": trial_row.episode,
                    "episode_simple": trial_row.episode[:-1],
                    "side": trial_row.side,
                    "loom_max_size": int(trial_row.loom_max_size),
                    "animal": animal_index + 1,
                    "baseline_dist": baseline_dist[animal_index],
                    "response_dist": response_dist[animal_index],
                    "response_over_baseline": ratio[animal_index],
                    "escape": bool(ratio[animal_index] > 2),
                }
            )

    return pd.DataFrame(rows)


def parse_animal_numbers(value) -> list[int]:
    """Parse metadata animal IDs stored as an inclusive range or a list."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop) + 1))
    return [int(item) for item in re.split(r"[\s,]+", text) if item]


def normalize_genotype(value) -> str | None:
    """Return the case-insensitive genotype key used by loom analyses."""
    if pd.isna(value):
        return None
    genotype = str(value).strip().lower()
    return genotype if genotype and genotype != "na" else None


def loom_genotype_catalog(meta_path: str | Path) -> list[str]:
    """List valid normalized genotype labels in the ``AllAn`` metadata sheet."""
    info_an = pd.read_excel(meta_path, sheet_name="AllAn")
    if "genotype" not in info_an:
        return []
    return sorted(
        {
            genotype
            for value in info_an["genotype"]
            if (genotype := normalize_genotype(value)) is not None
        }
    )


def resolve_requested_genotypes(requested, available) -> tuple[list[str], list[str]]:
    """Return requested genotypes present in the data and requested labels to skip."""
    normalized_requested = list(
        dict.fromkeys(
            genotype
            for value in requested
            if (genotype := normalize_genotype(value)) is not None
        )
    )
    normalized_available = {
        genotype
        for value in available
        if (genotype := normalize_genotype(value)) is not None
    }
    active = [value for value in normalized_requested if value in normalized_available]
    missing = [value for value in normalized_requested if value not in normalized_available]
    return active, missing


def select_loom_experiments(
    meta_path: str | Path,
    date_filter_mode="include_from_date",
    include_from_date="2026-01-01",
    exclude_date_start=None,
    exclude_date_end=None,
    include_lines=None,
    include_genotypes=None,
    include_line_sets=None,
    include_folders=None,
    exclude_folders=(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select raw shoaling experiments using metadata and date filters.

    Returns the usable experiments and a second table describing selected
    metadata rows whose ``PositionTxt*`` input could not be found.
    """
    info = pd.read_excel(meta_path, sheet_name="AllExp")
    info_an = pd.read_excel(meta_path, sheet_name="AllAn")
    info_an["expDate"] = pd.to_datetime(
        info_an["expDate"], format="%d-%m-%Y", dayfirst=True, errors="coerce"
    )

    date_filter = _validate_date_filter(
        date_filter_mode,
        include_from_date,
        exclude_date_start,
        exclude_date_end,
    )
    line_filter = _filter_set(include_lines)
    genotype_filter = (
        None
        if include_genotypes is None
        else {
            genotype
            for value in include_genotypes
            if (genotype := normalize_genotype(value)) is not None
        }
    )
    line_set_filter = _filter_set(include_line_sets)
    folder_filter = _filter_set(include_folders)
    excluded = _filter_set(exclude_folders) or set()

    records = []
    missing = []
    for metadata_index, row in info.iterrows():
        folder = str(row.get("folder", ""))
        if folder_filter is not None and folder not in folder_filter:
            continue
        if folder in excluded:
            continue

        animal_ids = parse_animal_numbers(row.get("anNr", ""))
        animals = info_an.loc[info_an["anNr"].isin(animal_ids)].copy()
        experiment_dates = pd.DatetimeIndex(animals["expDate"].dropna()).normalize()
        years = sorted(experiment_dates.year.astype(int).unique())
        lines = sorted(animals["line"].dropna().astype(str).unique()) if "line" in animals else []
        animal_genotypes = {}
        if "genotype" in animals:
            animal_genotypes = {
                int(animal.anNr): genotype
                for animal in animals[["anNr", "genotype"]].itertuples(index=False)
                if (genotype := normalize_genotype(animal.genotype)) is not None
            }
        genotypes = sorted(set(animal_genotypes.values()))
        line_sets = sorted({f"{line}_{row.get('date', '')}" for line in lines})

        if not _matches_date_filter(experiment_dates, date_filter):
            continue
        if not _intersects(lines, line_filter):
            continue
        if not _intersects(genotypes, genotype_filter):
            continue
        if not _intersects(line_sets, line_set_filter):
            continue

        start_dir = os.path.join(str(row.get("path", "")), folder)
        position_files = sorted(glob.glob(os.path.join(start_dir, "PositionTxt*")))
        if not position_files:
            missing.append(
                {
                    "metadata_index": int(metadata_index),
                    "folder": folder,
                    "start_dir": start_dir,
                    "reason": "No PositionTxt* file found",
                }
            )
            continue

        record = row.to_dict()
        record.update(
            {
                "metadata_index": int(metadata_index),
                "experiment": folder,
                "txt_path": position_files[0],
                "animal_ids": animal_ids,
                "animal_genotypes": animal_genotypes,
                "n_animals": len(animal_ids),
                "years": years,
                "exp_year": years[0] if len(years) == 1 else np.nan,
                "lines": lines,
                "genotypes": genotypes,
                "line_sets": line_sets,
            }
        )
        records.append(record)

    selected = pd.DataFrame(records).reset_index(drop=True)
    if not selected.empty:
        selected.insert(0, "experiment_index", np.arange(len(selected), dtype=int))
    return selected, pd.DataFrame(missing)


def add_condition_trial_index(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Add a one-based repeat index within each loom-size and side condition."""
    trials = trials_df.sort_values("block_start_frame").copy()
    trials["condition_trial"] = trials.groupby(
        ["loom_max_size", "side"], sort=False
    ).cumcount() + 1
    return trials.sort_values("trial").reset_index(drop=True)


def extract_trial_velocity_summary(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    frame_start: int = 0,
    frame_end: int = 300,
    step_frames: int = 30,
    fps: float = 30,
    units_per_mm: float = 4.0,
    n_animals: int = 35,
    animal_genotypes=None,
) -> pd.DataFrame:
    """Average trial velocity across all fish and, optionally, by genotype."""
    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError("Velocity frame range must satisfy 0 <= start < end.")
    if step_frames <= 0 or fps <= 0 or units_per_mm <= 0:
        raise ValueError("Velocity step, fps, and units_per_mm must be positive.")

    if units_per_mm <= 0:
        raise ValueError("units_per_mm must be positive.")
    x, y, orientation = animal_arrays(animal_df, n_animals=n_animals)
    genotype_groups = _genotype_index_groups(animal_genotypes, n_animals)
    sample_frames = np.arange(frame_start + step_frames, frame_end + 1, step_frames)
    seconds = step_frames / fps
    rows = []
    trials = _condition_indexed_trials(trials_df)

    for trial_row in trials.itertuples(index=False):
        block_start = int(trial_row.block_start_frame)
        block_end = int(trial_row.block_end_frame)
        start = block_start + frame_start
        stop = block_start + frame_end + 1
        if start < 0 or stop > len(animal_df) or stop - 1 > block_end:
            continue

        window_x = x[start:stop]
        window_y = y[start:stop]
        window_orientation = np.unwrap(orientation[start:stop], axis=0)
        for ep_frame in sample_frames:
            current = ep_frame - frame_start
            previous = current - step_frames
            linear = np.hypot(
                window_x[current] - window_x[previous],
                window_y[current] - window_y[previous],
            ) / units_per_mm / seconds
            angular = np.degrees(
                window_orientation[current] - window_orientation[previous]
            ) / seconds
            base_row = {
                    "trial": int(trial_row.trial),
                    "condition_trial": int(trial_row.condition_trial),
                    "block": int(trial_row.block),
                    "episode": trial_row.episode,
                    "side": trial_row.side,
                    "loom_max_size": int(trial_row.loom_max_size),
                    "epFrame": int(ep_frame),
            }
            if animal_genotypes is None:
                rows.append(
                    {
                        **base_row,
                        "linear_velocity_mm_s": float(np.nanmean(linear)),
                        "signed_angular_velocity_deg_s": float(np.nanmean(angular)),
                        "n_fish": int(np.isfinite(linear).sum()),
                    }
                )
                continue
            for genotype, indices in [(POOLED_GENOTYPE, np.arange(n_animals)), *genotype_groups.items()]:
                group_linear = linear[indices]
                group_angular = angular[indices]
                rows.append(
                    {
                        **base_row,
                        "genotype": genotype,
                        "linear_velocity_mm_s": float(np.nanmean(group_linear)),
                        "signed_angular_velocity_deg_s": float(np.nanmean(group_angular)),
                        "n_fish": int(np.isfinite(group_linear).sum()),
                    }
                )
    return pd.DataFrame(rows)


def trial_max_velocity_summary(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    onset_in_block: int = 150,
    window_frames: int = 120,
    step_frames: int = 30,
    fps: float = 30,
    units_per_mm: float = 4.0,
    n_animals: int = 35,
    animal_genotypes=None,
) -> pd.DataFrame:
    """Compute post-loom maximum speed across all fish and optionally by genotype."""
    trial_summary, _ = _trial_max_velocity_tables(
        animal_df,
        trials_df,
        onset_in_block=onset_in_block,
        window_frames=window_frames,
        step_frames=step_frames,
        fps=fps,
        units_per_mm=units_per_mm,
        n_animals=n_animals,
        animal_genotypes=animal_genotypes,
    )
    return trial_summary


def _trial_max_velocity_tables(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    onset_in_block: int = 150,
    window_frames: int = 120,
    pre_window_frames: int | None = None,
    step_frames: int = 30,
    fps: float = 30,
    units_per_mm: float = 4.0,
    n_animals: int = 35,
    animal_genotypes=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return trial summaries plus pre-mean and post-maximum speeds per fish."""
    if window_frames <= 0 or step_frames <= 0 or window_frames % step_frames:
        raise ValueError("window_frames must be a positive multiple of step_frames.")
    if pre_window_frames is not None and (
        pre_window_frames <= 0 or pre_window_frames % step_frames
    ):
        raise ValueError("pre_window_frames must be a positive multiple of step_frames.")
    if fps <= 0 or units_per_mm <= 0:
        raise ValueError("fps and units_per_mm must be positive.")

    x, y, _ = animal_arrays(animal_df, n_animals=n_animals)
    genotype_groups = _genotype_index_groups(animal_genotypes, n_animals)
    seconds = step_frames / fps
    post_sample_offsets = np.arange(step_frames, window_frames + 1, step_frames)
    pre_sample_offsets = (
        np.arange(-pre_window_frames + step_frames, 1, step_frames)
        if pre_window_frames is not None
        else None
    )
    rows = []
    fish_rows = []
    trials = _condition_indexed_trials(trials_df)

    for trial_row in trials.itertuples(index=False):
        onset = int(trial_row.block_start_frame) + onset_in_block
        post_sample_frames = onset + post_sample_offsets
        pre_sample_frames = (
            onset + pre_sample_offsets if pre_sample_offsets is not None else None
        )
        if (
            (pre_window_frames is not None and onset - pre_window_frames < 0)
            or post_sample_frames[-1] >= len(animal_df)
            or post_sample_frames[-1] > int(trial_row.block_end_frame)
        ):
            continue

        post_speed_windows = []
        for current in post_sample_frames:
            previous = int(current - step_frames)
            current = int(current)
            post_speed_windows.append(
                np.hypot(x[current] - x[previous], y[current] - y[previous])
                / units_per_mm
                / seconds
            )
        post_speed_stack = np.vstack(post_speed_windows)
        valid_fish = np.isfinite(post_speed_stack).any(axis=0)
        fish_max = np.full(n_animals, np.nan)
        if valid_fish.any():
            fish_max[valid_fish] = np.nanmax(post_speed_stack[:, valid_fish], axis=0)

        fish_pre_mean = np.full(n_animals, np.nan)
        if pre_sample_frames is not None:
            pre_speed_windows = []
            for current in pre_sample_frames:
                previous = int(current - step_frames)
                current = int(current)
                pre_speed_windows.append(
                    np.hypot(x[current] - x[previous], y[current] - y[previous])
                    / units_per_mm
                    / seconds
                )
            pre_speed_stack = np.vstack(pre_speed_windows)
            valid_pre_fish = np.isfinite(pre_speed_stack).any(axis=0)
            if valid_pre_fish.any():
                fish_pre_mean[valid_pre_fish] = np.nanmean(
                    pre_speed_stack[:, valid_pre_fish], axis=0
                )
        base_row = {
            "trial": int(trial_row.trial),
            "condition_trial": int(trial_row.condition_trial),
            "block": int(trial_row.block),
            "episode": trial_row.episode,
            "side": trial_row.side,
            "loom_max_size": int(trial_row.loom_max_size),
            "condition": f"{int(trial_row.loom_max_size)} {trial_row.side}",
            "fixed_loom_onset_frame": onset,
            "time_since_experiment_start_s": onset / fps,
            "time_since_experiment_start_min": onset / fps / 60,
        }
        for animal_index, (pre_mean, maximum) in enumerate(
            zip(fish_pre_mean, fish_max), start=1
        ):
            fish_rows.append(
                {
                    **base_row,
                    "animal": animal_index,
                    "fish_mean_pre_loom_linear_velocity_mm_s": pre_mean,
                    "fish_max_linear_velocity_mm_s": maximum,
                }
            )
        if animal_genotypes is None:
            rows.append(
                {
                    **base_row,
                    "trial_median_max_linear_velocity_mm_s": float(np.nanmedian(fish_max)),
                    "n_fish": int(np.isfinite(fish_max).sum()),
                }
            )
            continue
        for genotype, indices in [(POOLED_GENOTYPE, np.arange(n_animals)), *genotype_groups.items()]:
            group_max = fish_max[indices]
            rows.append(
                {
                    **base_row,
                    "genotype": genotype,
                    "trial_median_max_linear_velocity_mm_s": float(np.nanmedian(group_max)),
                    "n_fish": int(np.isfinite(group_max).sum()),
                }
            )
    fish_columns = [
        "trial",
        "condition_trial",
        "block",
        "episode",
        "side",
        "loom_max_size",
        "condition",
        "fixed_loom_onset_frame",
        "time_since_experiment_start_s",
        "time_since_experiment_start_min",
        "animal",
        "fish_mean_pre_loom_linear_velocity_mm_s",
        "fish_max_linear_velocity_mm_s",
    ]
    return pd.DataFrame(rows), pd.DataFrame(fish_rows, columns=fish_columns)


def collect_or_load_loom_analysis_data(
    experiments: pd.DataFrame,
    processing_dir: str | Path,
    force_reprocess: bool = False,
    onset_in_block: int = 150,
    pre_frames: int = 50,
    post_frames: int = 99,
    baseline_frames: tuple[int, int] = (100, 140),
    response_frames: tuple[int, int] = (149, 180),
    velocity_frame_start: int = 0,
    velocity_frame_end: int = 300,
    velocity_step_frames: int = 30,
    fps: float = 30,
    units_per_mm: float = 4.0,
    max_velocity_window_frames: int = 120,
    pre_velocity_window_frames: int = 120,
) -> dict[str, pd.DataFrame]:
    """Load cached compact loom tables or derive them one experiment at a time."""
    if experiments.empty:
        raise ValueError("No loom experiments were supplied.")
    processing_dir = Path(processing_dir)
    processing_dir.mkdir(parents=True, exist_ok=True)
    table_names = (
        "center_traces",
        "response_metrics",
        "trial_velocity",
        "trial_max_velocity",
        "fish_trial_max_velocity",
        "unsided_response_metrics",
        "unsided_fish_trial_max_velocity",
    )
    parts = {name: [] for name in table_names}
    valid_experiment_count = 0

    for row_number, row in experiments.reset_index(drop=True).iterrows():
        experiment = str(row.get("experiment", row.get("folder", row_number)))
        txt_path = Path(str(row["txt_path"]))
        n_animals = int(row.get("n_animals", 35))
        if n_animals <= 0:
            raise ValueError(f"{experiment}: metadata contains no animal IDs.")
        animal_ids = [int(value) for value in row.get("animal_ids", range(1, n_animals + 1))]
        raw_animal_genotypes = row.get("animal_genotypes", {})
        if not isinstance(raw_animal_genotypes, dict):
            raw_animal_genotypes = {}
        animal_genotype_map = {
            int(animal_id): normalize_genotype(genotype)
            for animal_id, genotype in raw_animal_genotypes.items()
            if normalize_genotype(genotype) is not None
        }
        animal_genotypes = [animal_genotype_map.get(animal_id) for animal_id in animal_ids]
        cache_paths = loom_cache_paths(row, processing_dir)
        settings = _loom_cache_settings(
            txt_path,
            n_animals,
            animal_ids,
            animal_genotypes,
            onset_in_block,
            pre_frames,
            post_frames,
            baseline_frames,
            response_frames,
            velocity_frame_start,
            velocity_frame_end,
            velocity_step_frames,
            fps,
            units_per_mm,
            max_velocity_window_frames,
            pre_velocity_window_frames,
        )
        cache_ready = all(cache_paths[name].exists() for name in table_names)
        cache_matches = cache_ready and _cache_settings_match(cache_paths["settings"], settings)

        if cache_matches and not force_reprocess:
            print(f"Loading loom cache {row_number + 1}/{len(experiments)}: {experiment}", flush=True)
            tables = {name: pd.read_csv(cache_paths[name]) for name in table_names}
        else:
            reason = "forced" if force_reprocess else "missing or stale cache"
            print(
                f"Processing loom experiment {row_number + 1}/{len(experiments)}: "
                f"{experiment} ({reason})",
                flush=True,
            )
            try:
                tables = _process_loom_experiment(
                    row,
                    onset_in_block,
                    pre_frames,
                    post_frames,
                    baseline_frames,
                    response_frames,
                    velocity_frame_start,
                    velocity_frame_end,
                    velocity_step_frames,
                    fps,
                    units_per_mm,
                    max_velocity_window_frames,
                    pre_velocity_window_frames,
                )
            except _NoLoomStimulusEpisodesError as error:
                warnings.warn(
                    f"Skipping loom experiment {experiment}: {error}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            for name in table_names:
                tables[name].to_csv(cache_paths[name], index=False, compression="gzip")
            with cache_paths["settings"].open("w", encoding="utf-8") as stream:
                json.dump(settings, stream, indent=2, sort_keys=True)

        for name in table_names:
            parts[name].append(tables[name])
        valid_experiment_count += 1

    if valid_experiment_count == 0:
        raise ValueError(
            "No usable loom experiments remain after skipping experiments without supported "
            "CLfull or CLsemi episodes."
        )

    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in parts.items()
    }


def summarize_across_experiments(
    frame: pd.DataFrame,
    group_columns: list[str] | tuple[str, ...],
    value_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return equal-experiment means, SEMs, and contributing experiment counts."""
    group_columns = list(group_columns)
    value_columns = list(value_columns)
    required = {"experiment", *group_columns, *value_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for experiment summary: {missing}")

    experiment_values = frame.groupby(
        ["experiment", *group_columns], as_index=False, dropna=False
    )[value_columns].mean()
    grouped = experiment_values.groupby(group_columns, dropna=False)[value_columns]
    mean = grouped.mean().add_suffix("_mean").reset_index()
    sem = grouped.sem().add_suffix("_sem").reset_index()
    counts = grouped.count().max(axis=1).rename("n_experiments").reset_index()
    return mean.merge(sem, on=group_columns).merge(counts, on=group_columns)


def loom_cache_paths(row: pd.Series, processing_dir: str | Path) -> dict[str, Path]:
    """Return stable per-experiment cache paths for the compact loom tables."""
    experiment = row.get("experiment", row.get("folder", "experiment"))
    stem = Path(str(row["txt_path"])).stem
    label = _safe_filename(f"{experiment}_{stem}")
    root = Path(processing_dir)
    return {
        "center_traces": root / f"{label}_center_traces.csv.gz",
        "response_metrics": root / f"{label}_response_metrics.csv.gz",
        "trial_velocity": root / f"{label}_trial_velocity.csv.gz",
        "trial_max_velocity": root / f"{label}_trial_max_velocity.csv.gz",
        "fish_trial_max_velocity": root / f"{label}_fish_trial_max_velocity.csv.gz",
        "unsided_response_metrics": root / f"{label}_unsided_response_metrics.csv.gz",
        "unsided_fish_trial_max_velocity": root / f"{label}_unsided_fish_trial_max_velocity.csv.gz",
        "settings": root / f"{label}_settings.json",
    }


def _process_loom_experiment(
    row,
    onset_in_block,
    pre_frames,
    post_frames,
    baseline_frames,
    response_frames,
    velocity_frame_start,
    velocity_frame_end,
    velocity_step_frames,
    fps,
    units_per_mm,
    max_velocity_window_frames,
    pre_velocity_window_frames=120,
):
    experiment = str(row.get("experiment", row.get("folder", "experiment")))
    experiment_index = int(row.get("experiment_index", 0))
    animal_ids = [int(value) for value in row["animal_ids"]]
    n_animals = len(animal_ids)
    animals = load_animal_file(row["txt_path"], n_animals=n_animals)
    stimulus = embedded_stimulus_from_animal_file(animals)
    all_block_trials = loom_trials(stimulus)
    unsided_trials = unsided_loom_trials(stimulus, onset_in_block=onset_in_block)
    if all_block_trials.empty and unsided_trials.empty:
        raise _NoLoomStimulusEpisodesError(
            "no supported CLfull or CLsemi loom stimulus episodes were found."
        )
    # Existing notebook analyses intentionally receive only L/R-labelled looms.
    trials = all_block_trials.loc[
        all_block_trials["episode"].astype(str).str.fullmatch(SUPPORTED_LOOM_EPISODE_PATTERN)
    ].copy()
    if trials.empty:
        raise _NoLoomStimulusEpisodesError(
            "no sided CLfull or CLsemi loom stimulus episodes were found."
        )
    trials = add_condition_trial_index(trials)

    snippets = extract_legacy_loom_snippets(
        animals,
        trials,
        stimulus,
        pre_frames=pre_frames,
        post_frames=post_frames,
        onset_in_block=onset_in_block,
        n_animals=n_animals,
        units_per_mm=units_per_mm,
    )
    snippets["animal_id"] = snippets["animal"].map(
        {index + 1: animal_id for index, animal_id in enumerate(animal_ids)}
    )
    raw_animal_genotypes = row.get("animal_genotypes", {})
    if not isinstance(raw_animal_genotypes, dict):
        raw_animal_genotypes = {}
    animal_genotype_map = {
        int(animal_id): normalize_genotype(genotype)
        for animal_id, genotype in raw_animal_genotypes.items()
        if normalize_genotype(genotype) is not None
    }
    animal_genotypes = [animal_genotype_map.get(animal_id) for animal_id in animal_ids]
    snippets["genotype"] = snippets["animal_id"].map(animal_genotype_map)
    animal_traces = snippets.groupby(
        ["loom_max_size", "side", "animal_id", "genotype", "relative_frame"],
        as_index=False,
        dropna=False,
    )["center_dist_mm"].mean()
    pooled_center = animal_traces.groupby(
        ["loom_max_size", "side", "relative_frame"], as_index=False
    ).agg(center_dist_mm=("center_dist_mm", "mean"), n_animals=("animal_id", "nunique"))
    pooled_center["genotype"] = POOLED_GENOTYPE
    genotype_center = animal_traces.dropna(subset=["genotype"]).groupby(
        ["loom_max_size", "side", "genotype", "relative_frame"], as_index=False
    ).agg(center_dist_mm=("center_dist_mm", "mean"), n_animals=("animal_id", "nunique"))
    center_traces = pd.concat([pooled_center, genotype_center], ignore_index=True)

    raw_metrics = legacy_loom_response_metrics(
        animals,
        trials,
        baseline_frames=tuple(baseline_frames),
        response_frames=tuple(response_frames),
        n_animals=n_animals,
    )
    raw_metrics["animal_id"] = raw_metrics["animal"].map(
        {index + 1: animal_id for index, animal_id in enumerate(animal_ids)}
    )
    raw_metrics["genotype"] = raw_metrics["animal_id"].map(animal_genotype_map)
    metric_columns = ["escape", "response_dist", "response_over_baseline"]
    # Retain the fish-by-trial grain so notebooks can choose how to combine
    # sides and repeats without reconstructing metrics from raw trajectories.
    animal_trial_metrics = raw_metrics[
        ["trial", "episode", "loom_max_size", "side", "animal_id", "genotype", *metric_columns]
    ].copy()
    animal_trial_metrics["n_animals"] = 1
    animal_trial_metrics["aggregation"] = "animal_trial"
    animal_side = raw_metrics.groupby(
        ["loom_max_size", "side", "animal_id", "genotype"], as_index=False, dropna=False
    )[metric_columns].mean()
    pooled_side_metrics = animal_side.groupby(["loom_max_size", "side"], as_index=False).agg(
        escape=("escape", "mean"),
        response_dist=("response_dist", "mean"),
        response_over_baseline=("response_over_baseline", "mean"),
        n_animals=("animal_id", "nunique"),
    )
    pooled_side_metrics["genotype"] = POOLED_GENOTYPE
    genotype_side_metrics = animal_side.dropna(subset=["genotype"]).groupby(
        ["loom_max_size", "side", "genotype"], as_index=False
    ).agg(
        escape=("escape", "mean"),
        response_dist=("response_dist", "mean"),
        response_over_baseline=("response_over_baseline", "mean"),
        n_animals=("animal_id", "nunique"),
    )
    side_metrics = pd.concat([pooled_side_metrics, genotype_side_metrics], ignore_index=True)
    side_metrics["aggregation"] = "side_resolved"
    animal_merged = raw_metrics.groupby(
        ["loom_max_size", "animal_id", "genotype"], as_index=False, dropna=False
    )[metric_columns].mean()
    pooled_merged_metrics = animal_merged.groupby("loom_max_size", as_index=False).agg(
        escape=("escape", "mean"),
        response_dist=("response_dist", "mean"),
        response_over_baseline=("response_over_baseline", "mean"),
        n_animals=("animal_id", "nunique"),
    )
    pooled_merged_metrics["genotype"] = POOLED_GENOTYPE
    genotype_merged_metrics = animal_merged.dropna(subset=["genotype"]).groupby(
        ["loom_max_size", "genotype"], as_index=False
    ).agg(
        escape=("escape", "mean"),
        response_dist=("response_dist", "mean"),
        response_over_baseline=("response_over_baseline", "mean"),
        n_animals=("animal_id", "nunique"),
    )
    merged_metrics = pd.concat([pooled_merged_metrics, genotype_merged_metrics], ignore_index=True)
    merged_metrics["side"] = "merged"
    merged_metrics["aggregation"] = "lr_merged"
    response_metrics = pd.concat(
        [side_metrics, merged_metrics, animal_trial_metrics], ignore_index=True
    )

    trial_velocity = extract_trial_velocity_summary(
        animals,
        trials,
        frame_start=velocity_frame_start,
        frame_end=velocity_frame_end,
        step_frames=velocity_step_frames,
        fps=fps,
        units_per_mm=units_per_mm,
        n_animals=n_animals,
        animal_genotypes=animal_genotypes,
    )
    trial_max, fish_trial_max = _trial_max_velocity_tables(
        animals,
        trials,
        onset_in_block=onset_in_block,
        window_frames=max_velocity_window_frames,
        pre_window_frames=pre_velocity_window_frames,
        step_frames=velocity_step_frames,
        fps=fps,
        units_per_mm=units_per_mm,
        n_animals=n_animals,
        animal_genotypes=animal_genotypes,
    )
    fish_trial_max["animal_id"] = fish_trial_max["animal"].map(
        {index + 1: animal_id for index, animal_id in enumerate(animal_ids)}
    )
    fish_trial_max["genotype"] = fish_trial_max["animal_id"].map(animal_genotype_map)

    unsided_response_metrics = pd.DataFrame(
        columns=[
            "trial",
            "condition_trial",
            "episode",
            "loom_max_size",
            "side",
            "animal_id",
            "genotype",
            *metric_columns,
            "n_animals",
            "aggregation",
        ]
    )
    unsided_fish_trial_max = fish_trial_max.iloc[0:0].copy()
    if not unsided_trials.empty:
        unsided_trials = add_condition_trial_index(unsided_trials)
        unsided_raw_metrics = legacy_loom_response_metrics(
            animals,
            unsided_trials,
            baseline_frames=tuple(baseline_frames),
            response_frames=tuple(response_frames),
            n_animals=n_animals,
        )
        unsided_raw_metrics["animal_id"] = unsided_raw_metrics["animal"].map(
            {index + 1: animal_id for index, animal_id in enumerate(animal_ids)}
        )
        unsided_raw_metrics["genotype"] = unsided_raw_metrics["animal_id"].map(
            animal_genotype_map
        )
        unsided_response_metrics = unsided_raw_metrics[
            [
                "trial",
                "episode",
                "loom_max_size",
                "side",
                "animal_id",
                "genotype",
                *metric_columns,
            ]
        ].merge(
            unsided_trials[["trial", "condition_trial"]],
            on="trial",
            how="left",
            validate="many_to_one",
        )
        unsided_response_metrics["n_animals"] = 1
        unsided_response_metrics["aggregation"] = "animal_trial"

        _, unsided_fish_trial_max = _trial_max_velocity_tables(
            animals,
            unsided_trials,
            onset_in_block=onset_in_block,
            window_frames=max_velocity_window_frames,
            pre_window_frames=pre_velocity_window_frames,
            step_frames=velocity_step_frames,
            fps=fps,
            units_per_mm=units_per_mm,
            n_animals=n_animals,
            animal_genotypes=animal_genotypes,
        )
        unsided_fish_trial_max["animal_id"] = unsided_fish_trial_max["animal"].map(
            {index + 1: animal_id for index, animal_id in enumerate(animal_ids)}
        )
        unsided_fish_trial_max["genotype"] = unsided_fish_trial_max["animal_id"].map(
            animal_genotype_map
        )

    for table in (
        center_traces,
        response_metrics,
        trial_velocity,
        trial_max,
        fish_trial_max,
        unsided_response_metrics,
        unsided_fish_trial_max,
    ):
        table.insert(0, "experiment", experiment)
        table.insert(0, "experiment_index", experiment_index)
        table["txt_path"] = str(row["txt_path"])
    return {
        "center_traces": center_traces,
        "response_metrics": response_metrics,
        "trial_velocity": trial_velocity,
        "trial_max_velocity": trial_max,
        "fish_trial_max_velocity": fish_trial_max,
        "unsided_response_metrics": unsided_response_metrics,
        "unsided_fish_trial_max_velocity": unsided_fish_trial_max,
    }


def _loom_cache_settings(
    txt_path,
    n_animals,
    animal_ids,
    animal_genotypes,
    onset_in_block,
    pre_frames,
    post_frames,
    baseline_frames,
    response_frames,
    velocity_frame_start,
    velocity_frame_end,
    velocity_step_frames,
    fps,
    units_per_mm,
    max_velocity_window_frames,
    pre_velocity_window_frames,
):
    stat = txt_path.stat()
    return {
        "schema_version": LOOM_CACHE_SCHEMA_VERSION,
        "txt_path": str(txt_path.resolve()),
        "txt_size": int(stat.st_size),
        "txt_mtime_ns": int(stat.st_mtime_ns),
        "n_animals": int(n_animals),
        "animal_ids": [int(value) for value in animal_ids],
        "animal_genotypes": [value if value is not None else None for value in animal_genotypes],
        "onset_in_block": int(onset_in_block),
        "pre_frames": int(pre_frames),
        "post_frames": int(post_frames),
        "baseline_frames": [int(value) for value in baseline_frames],
        "response_frames": [int(value) for value in response_frames],
        "velocity_frame_start": int(velocity_frame_start),
        "velocity_frame_end": int(velocity_frame_end),
        "velocity_step_frames": int(velocity_step_frames),
        "fps": float(fps),
        "units_per_mm": float(units_per_mm),
        "max_velocity_window_frames": int(max_velocity_window_frames),
        "pre_velocity_window_frames": int(pre_velocity_window_frames),
    }


def _cache_settings_match(settings_path: Path, expected: dict) -> bool:
    if not settings_path.exists():
        return False
    try:
        with settings_path.open("r", encoding="utf-8") as stream:
            return json.load(stream) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _condition_indexed_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    if "condition_trial" in trials_df:
        return trials_df.copy()
    return add_condition_trial_index(trials_df)


def _genotype_index_groups(animal_genotypes, n_animals: int) -> dict[str, np.ndarray]:
    if animal_genotypes is None:
        return {}
    if len(animal_genotypes) != n_animals:
        raise ValueError("animal_genotypes must have one entry per tracked fish.")
    normalized = np.asarray([normalize_genotype(value) for value in animal_genotypes], dtype=object)
    return {
        genotype: np.flatnonzero(normalized == genotype)
        for genotype in sorted({value for value in normalized if value is not None})
    }


def _filter_set(values):
    if values is None:
        return None
    return set(values)


def _intersects(values, allowed) -> bool:
    return allowed is None or bool(set(values) & allowed)


def _parse_user_date(value, setting_name: str) -> pd.Timestamp:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{setting_name} must be an ISO date in YYYY-MM-DD format.")
    try:
        parsed = pd.Timestamp(value).normalize()
    except (TypeError, ValueError):
        raise ValueError(f"{setting_name} must be an ISO date in YYYY-MM-DD format.") from None
    if pd.isna(parsed):
        raise ValueError(f"{setting_name} must be an ISO date in YYYY-MM-DD format.")
    return parsed


def _validate_date_filter(
    mode: str,
    include_from_date,
    exclude_date_start,
    exclude_date_end,
) -> dict[str, object]:
    valid_modes = {"include_from_date", "exclude_date_range"}
    if mode not in valid_modes:
        raise ValueError(f"date_filter_mode must be one of {sorted(valid_modes)}; got {mode!r}.")
    if mode == "include_from_date":
        return {
            "mode": mode,
            "start": _parse_user_date(include_from_date, "include_from_date"),
        }

    start = _parse_user_date(exclude_date_start, "exclude_date_start")
    end = _parse_user_date(exclude_date_end, "exclude_date_end")
    if start > end:
        raise ValueError("exclude_date_start must be on or before exclude_date_end.")
    return {"mode": mode, "start": start, "end": end}


def _matches_date_filter(experiment_dates, date_filter: dict[str, object]) -> bool:
    dates = pd.DatetimeIndex(experiment_dates).dropna().normalize()
    if dates.empty:
        return False
    if date_filter["mode"] == "include_from_date":
        return bool((dates >= date_filter["start"]).any())
    return not bool(
        ((dates >= date_filter["start"]) & (dates <= date_filter["end"])).any()
    )


def _safe_filename(value) -> str:
    text = str(value).strip()
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in text)


def extract_grating_snippets(
    animal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    pre_frames: int = 100,
    post_frames: int = 899,
    n_animals: int = 35,
    north_angle: float = np.pi / 2,
) -> pd.DataFrame:
    x, y, orientation = animal_arrays(animal_df, n_animals=n_animals)
    relative_frames = np.arange(-pre_frames, post_frames + 1)
    snippets = []

    for trial_row in trials_df.itertuples(index=False):
        onset = int(trial_row.block_start_frame)
        start = onset - pre_frames
        stop = onset + post_frames + 1
        if start < 0 or stop > len(animal_df):
            continue

        window_x = x[start:stop]
        window_y = y[start:stop]
        onset_x = x[onset]
        onset_y = y[onset]
        onset_orientation = orientation[onset]

        raw_dx = window_x - onset_x
        raw_dy = window_y - onset_y
        dy = -raw_dy

        heading_angle = -onset_orientation
        rotation = north_angle - heading_angle
        cos_rotation = np.cos(rotation)
        sin_rotation = np.sin(rotation)

        ego_x = raw_dx * cos_rotation - dy * sin_rotation
        ego_y = raw_dx * sin_rotation + dy * cos_rotation
        distance = np.hypot(ego_x, ego_y)

        for animal_index in range(n_animals):
            snippets.append(
                pd.DataFrame(
                    {
                        "trial": int(trial_row.trial),
                        "block": int(trial_row.block),
                        "episode": trial_row.episode,
                        "direction": trial_row.direction,
                        "frequency_code": int(trial_row.frequency_code),
                        "animal": animal_index + 1,
                        "relative_frame": relative_frames,
                        "global_frame": np.arange(start, stop),
                        "raw_dx": raw_dx[:, animal_index],
                        "raw_dy": raw_dy[:, animal_index],
                        "ego_x": ego_x[:, animal_index],
                        "ego_y": ego_y[:, animal_index],
                        "distance_from_onset": distance[:, animal_index],
                        "raw_x": window_x[:, animal_index],
                        "raw_y": window_y[:, animal_index],
                        "orientation_at_onset": onset_orientation[animal_index],
                    }
                )
            )

    return pd.concat(snippets, ignore_index=True)
