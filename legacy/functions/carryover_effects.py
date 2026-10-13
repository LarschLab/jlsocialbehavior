import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Episodes used in the carryover analysis; keys are raw protocol labels and values are plot labels.
DEFAULT_EPISODE_MAP = {
    "01k01f": "continuous",
    "02k20f": "bouts",
}


def normalize_episode_label(value):
    # Bonsai episode labels sometimes contain underscores or whitespace; remove those before matching.
    return str(value).strip().replace("_", "")


def parse_an_nrs(value):
    # Metadata stores animal IDs either as "101:135" ranges or as explicit comma/space-separated lists.
    text = str(value).strip()
    if ":" in text:
        start, stop = text.split(":", 1)
        return np.arange(int(start), int(stop) + 1)
    return np.array(text.replace(",", " ").split()).astype(int)


def first_match(pattern):
    # Fail loudly when expected raw inputs are missing so notebook output is not silently incomplete.
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def read_selection_metadata(meta_path, year=None, drop_na_genotype=True):
    # Load experiment-level rows and animal-level annotations from the shared selection metadata workbook.
    info = pd.read_excel(meta_path, sheet_name="AllExp")
    info_an = pd.read_excel(meta_path, sheet_name="AllAn")

    # Convert dates once here so downstream filters and labels do not repeat parsing logic.
    info["date_dt"] = pd.to_datetime(info["date"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
    info["date_label"] = info["date_dt"].dt.strftime("%d-%m-%Y")
    info_an["bd"] = pd.to_datetime(info_an["bd"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
    if "expDate" in info_an:
        info_an["expDate"] = pd.to_datetime(info_an["expDate"], format="%d-%m-%Y", dayfirst=True, errors="coerce")

    # The plotting notebook mirrors current selection notebooks by excluding literal "na" genotypes.
    if drop_na_genotype and "genotype" in info_an:
        info_an = info_an.loc[
            info_an["genotype"].notna()
            & info_an["genotype"].astype(str).str.lower().ne("na")
        ].copy()

    if year is not None:
        info = info.loc[info["date_dt"].dt.year == year].copy()

    return info.reset_index(drop=True), info_an.reset_index(drop=True)


def prepare_carryover_effect_table(info_source, episode_duration_min=5, in_dish=10, arena_diameter_mm=70):
    # Create a compact raw-analysis table from metadata without invoking the canonical 5 min writer.
    rows = []
    animal_lookup = {}

    for _, row in info_source.iterrows():
        # Each experiment folder should contain one PositionTxt file and one pair-list file.
        start_dir = os.path.join(str(row["path"]), str(row["folder"]))
        if not os.path.exists(start_dir):
            print("WARNING: path does not exist:", start_dir)
            continue

        # Keep pairList in the table for traceability, even though ASD uses the stimulus columns directly.
        pos_path = first_match(os.path.join(start_dir, "PositionTxt*"))
        pair_list = first_match(os.path.join(start_dir, "PL*"))
        an_ids = parse_an_nrs(row["anNr"])
        animal_set = len(rows)

        # These fields mirror the metadata used by the existing pipeline where that is useful for scale/time.
        out = row.copy()
        out["txtPath"] = pos_path
        out["pairList"] = pair_list
        out["epiDur"] = episode_duration_min
        out["inDish"] = in_dish
        out["arenaDiameter_mm"] = arena_diameter_mm
        out["anIDAll"] = " ".join(str(x) for x in an_ids)
        out["set"] = animal_set

        rows.append(out)
        animal_lookup[animal_set] = an_ids

    return pd.DataFrame(rows).reset_index(drop=True), animal_lookup


def load_raw_position_table(txt_path, read_lim=None):
    # Reproduce the relevant raw-loading branches from models.experiment without creating experiment outputs.
    txt_path = str(txt_path)
    read_lim = _clean_read_lim(read_lim)
    if txt_path[-3:] == "npy":
        # idtrackerai arrays store xy only; insert zero headings to match the pipeline's 3-column layout.
        raw = np.load(txt_path)
        tmp = np.dstack([raw, np.zeros(raw.shape[:2])])
        raw_data = pd.DataFrame(tmp.reshape((tmp.shape[0], 3 * raw.shape[1])))
        return raw_data, np.zeros(raw_data.shape[0])

    # The first character distinguishes older Bonsai, idTracker CSV, and current VR-tracker text formats.
    first_line = pd.read_csv(txt_path, header=None, nrows=1, sep=":")
    first_value = str(first_line.values[0][0])
    first_char = first_value[0] if len(first_value) > 0 else ""

    if first_char == "(":
        # Old Bonsai files interleave animal coordinates and episode labels differently from current VR files.
        raw_data = pd.read_csv(
            txt_path,
            engine="python",
            index_col=None,
            header=None,
            skipfooter=1,
            usecols=[2, 3, 6, 7, 10],
            names=np.arange(5),
        )
        episode_all = raw_data[4]
        raw_data = raw_data.drop(raw_data.columns[[4]], axis=1).astype(float)
        raw_data.insert(loc=2, column="o1", value=0)
        raw_data.insert(loc=5, column="o2", value=0)
        return raw_data, np.array(episode_all)

    if first_char == "X":
        # idTracker CSVs have a header row and no explicit episode labels.
        raw_data = pd.read_csv(txt_path, header=None, delim_whitespace=True, skiprows=1, nrows=read_lim)
        return raw_data, np.zeros(raw_data.shape[0])

    # Current Bonsai VR tracking files store the episode label in the last column.
    raw_data = pd.read_csv(txt_path, header=None, delim_whitespace=True, nrows=read_lim)
    episode_all = np.array(raw_data.loc[:, raw_data.columns[-1]])
    return raw_data, episode_all


def px_per_mm_from_row(row):
    # Prefer explicit metadata scale if present; otherwise infer it from the same ROI files as the pipeline.
    if "pxPmm" in row and pd.notna(row["pxPmm"]):
        return float(row["pxPmm"])

    start_dir = os.path.split(str(row["txtPath"]))[0]
    roi_paths = glob.glob(os.path.join(start_dir, "ROIdef*"))
    if roi_paths:
        # Skype/Bonsai ROI files store radius in the last column.
        rois = _load_data_flex_delim(roi_paths[0])
        radius_px = rois.mean(axis=0)[-1]
        return 2 * radius_px / float(row.get("arenaDiameter_mm", 70))

    roi_paths = glob.glob(os.path.join(start_dir, "bgMed_scale*"))
    if not roi_paths:
        roi_paths = glob.glob(os.path.join(start_dir, "*bgMed.csv"))
    if roi_paths:
        # idTracker-style ROI files store radius in column index 3 after the CSV header.
        rois = np.loadtxt(roi_paths[0], skiprows=1, delimiter=",")
        radius_px = rois.mean(axis=0)[3]
        return 2 * radius_px / float(row.get("arenaDiameter_mm", 70))

    raise FileNotFoundError("No ROI or pxPmm value found for " + str(row["txtPath"]))


def collect_carryover_effect_data(
    exp_rows,
    animal_info=None,
    animal_lookup=None,
    episode_map=None,
    max_episode_blocks=24,
    window_seconds=60,
    shuffle_min_shift_seconds=30,
    n_shift_runs=10,
    frame_stride=1,
    include_escapees=False,
):
    # Normalize episode labels once and build a fast animal-ID to metadata lookup.
    episode_map = _normalized_episode_map(episode_map or DEFAULT_EPISODE_MAP)
    animal_meta = _animal_metadata_lookup(animal_info)
    asd_parts = []
    si_parts = []
    exp_rows = exp_rows.reset_index(drop=True)
    total_experiments = len(exp_rows)

    for row_idx, row in exp_rows.iterrows():
        asd_part, si_part = _collect_carryover_effect_row(
            row,
            row_idx,
            total_experiments,
            animal_meta,
            animal_lookup,
            episode_map,
            max_episode_blocks,
            window_seconds,
            shuffle_min_shift_seconds,
            n_shift_runs,
            frame_stride,
            include_escapees,
        )
        asd_parts.append(asd_part)
        si_parts.append(si_part)

    asd = pd.concat(asd_parts, ignore_index=True) if asd_parts else pd.DataFrame()
    si = pd.concat(si_parts, ignore_index=True) if si_parts else pd.DataFrame()
    return asd, si


def collect_or_load_carryover_effect_data(
    exp_rows,
    processing_dir,
    animal_info=None,
    animal_lookup=None,
    episode_map=None,
    max_episode_blocks=24,
    window_seconds=60,
    shuffle_min_shift_seconds=30,
    n_shift_runs=10,
    frame_stride=1,
    include_escapees=False,
    force_reprocess=False,
    cache_settings=None,
):
    # Mirrors experiment_set(MissingOnly=True): reuse per-experiment processed tables when both exist.
    processing_dir = Path(processing_dir)
    processing_dir.mkdir(parents=True, exist_ok=True)
    episode_map = _normalized_episode_map(episode_map or DEFAULT_EPISODE_MAP)
    animal_meta = _animal_metadata_lookup(animal_info)
    asd_parts = []
    si_parts = []
    exp_rows = exp_rows.reset_index(drop=True)
    total_experiments = len(exp_rows)

    for row_idx, row in exp_rows.iterrows():
        cache_paths = carryover_cache_paths(row, processing_dir, row_idx)
        cache_ready = cache_paths["asd_frame"].exists() and cache_paths["si_1min"].exists()
        if cache_ready and not force_reprocess:
            _warn_if_cache_settings_mismatch(cache_paths["settings"], cache_settings, row, row_idx)
            print(
                f"Loading cached carryover tables {row_idx + 1}/{total_experiments}: "
                f"{_experiment_key(row, row_idx)}",
                flush=True,
            )
            asd_parts.append(pd.read_csv(cache_paths["asd_frame"]))
            si_parts.append(pd.read_csv(cache_paths["si_1min"]))
            continue

        asd_part, si_part = _collect_carryover_effect_row(
            row,
            row_idx,
            total_experiments,
            animal_meta,
            animal_lookup,
            episode_map,
            max_episode_blocks,
            window_seconds,
            shuffle_min_shift_seconds,
            n_shift_runs,
            frame_stride,
            include_escapees,
        )
        asd_part.to_csv(cache_paths["asd_frame"], index=False, compression="gzip")
        si_part.to_csv(cache_paths["si_1min"], index=False, compression="gzip")
        _write_cache_settings(cache_paths["settings"], cache_settings)
        print(f"  Cached ASD table: {cache_paths['asd_frame']}", flush=True)
        print(f"  Cached SI table: {cache_paths['si_1min']}", flush=True)
        asd_parts.append(asd_part)
        si_parts.append(si_part)

    asd = pd.concat(asd_parts, ignore_index=True) if asd_parts else pd.DataFrame()
    si = pd.concat(si_parts, ignore_index=True) if si_parts else pd.DataFrame()
    return asd, si


def carryover_cache_paths(row, processing_dir, row_idx=0):
    label = _safe_filename(
        f"{int(row_idx):03d}_{_experiment_key(row, row_idx)}_{Path(str(row['txtPath'])).stem}"
    )
    processing_dir = Path(processing_dir)
    return {
        "asd_frame": processing_dir / f"{label}_asd_frame.csv.gz",
        "si_1min": processing_dir / f"{label}_si_1min.csv.gz",
        "settings": processing_dir / f"{label}_settings.json",
    }


def _collect_carryover_effect_row(
    row,
    row_idx,
    total_experiments,
    animal_meta,
    animal_lookup,
    episode_map,
    max_episode_blocks,
    window_seconds,
    shuffle_min_shift_seconds,
    n_shift_runs,
    frame_stride,
    include_escapees,
):
    # Work experiment-by-experiment to avoid holding multiple raw position tables at once.
    experiment_label = _experiment_key(row, row_idx)
    print(f"Processing experiment {row_idx + 1}/{total_experiments}: {experiment_label}", flush=True)
    print(f"  Raw position file: {row['txtPath']}", flush=True)
    raw_data, episode_all = load_raw_position_table(row["txtPath"], row.get("readLim", None))
    fps = int(row.get("fps", 30)) if pd.notna(row.get("fps", 30)) else 30
    px_pmm = px_per_mm_from_row(row)
    n_animals = _infer_focal_animal_count(raw_data)
    stim_index = n_animals
    local_to_animal_id = _local_animal_ids(row, animal_lookup, row_idx, n_animals)
    blocks = _episode_blocks(episode_all, episode_map, max_episode_blocks)
    print(
        "  Loaded "
        f"{n_animals} focal animals; selected {len(blocks)} episode blocks; "
        f"include_escapees={include_escapees}",
        flush=True,
    )

    # In these files the stimulus is represented as the extra animal after all focal animals.
    asd_parts = []
    si_parts = []
    stim_xy = raw_data.iloc[:, [stim_index * 3, stim_index * 3 + 1]].to_numpy(dtype=float)
    included_animals = 0
    skipped_escapees = 0
    for local_index in range(n_animals):
        animal_id = int(local_to_animal_id[local_index])
        meta = animal_meta.get(animal_id, {})
        if not include_escapees and _is_escapee(meta):
            skipped_escapees += 1
            continue
        included_animals += 1
        animal_xy = raw_data.iloc[:, [local_index * 3, local_index * 3 + 1]].to_numpy(dtype=float)
        asd_mm = np.linalg.norm(animal_xy - stim_xy, axis=1) / px_pmm

        # Store dense frame-level ASD and 1-minute SI separately; the notebook chooses how to summarize them.
        asd_parts.append(
            _animal_frame_asd(
                asd_mm,
                blocks,
                row,
                row_idx,
                local_index,
                animal_id,
                meta,
                fps,
                frame_stride,
            )
        )
        si_parts.append(
            _animal_one_min_si(
                animal_xy,
                stim_xy,
                blocks,
                row,
                row_idx,
                local_index,
                animal_id,
                meta,
                fps,
                px_pmm,
                window_seconds,
                shuffle_min_shift_seconds,
                n_shift_runs,
            )
        )
    print(
        f"  Included {included_animals} animals"
        + (f"; skipped {skipped_escapees} escapee animals" if skipped_escapees else ""),
        flush=True,
    )
    asd = pd.concat(asd_parts, ignore_index=True) if asd_parts else pd.DataFrame()
    si = pd.concat(si_parts, ignore_index=True) if si_parts else pd.DataFrame()
    return asd, si


def _normalized_episode_map(episode_map):
    # Accept episode codes with or without the trailing "f" used in current metadata.
    out = {}
    for code, label in episode_map.items():
        key = normalize_episode_label(code)
        out[key] = label
        if not key.endswith("f"):
            out[key + "f"] = label
    return out


def _load_data_flex_delim(path):
    # ROI files in this repository may be whitespace-delimited or comma-delimited.
    try:
        return np.loadtxt(path)
    except Exception:
        return np.loadtxt(path, delimiter=",")


def _clean_read_lim(read_lim):
    # Pandas may pass missing numeric values as NaN; convert those to the nrows=None convention.
    if read_lim is None:
        return None
    try:
        if pd.isna(read_lim):
            return None
    except TypeError:
        return read_lim
    return int(read_lim)


def _animal_metadata_lookup(animal_info):
    # Return a plain dictionary so inner loops do not repeatedly query a DataFrame.
    if animal_info is None or animal_info.empty or "anNr" not in animal_info:
        return {}
    return animal_info.set_index("anNr").to_dict(orient="index")


def _infer_focal_animal_count(raw_data):
    # Raw tables use 3 columns per animal plus 3 stimulus columns and 1 episode/stim-size column.
    n_cols = raw_data.shape[1]
    inferred = (n_cols - 4) / 3
    if inferred == int(inferred) and inferred > 0:
        return int(inferred)
    raise ValueError("Could not infer focal animal count from raw data shape " + str(raw_data.shape))


def _local_animal_ids(row, animal_lookup, row_idx, n_animals):
    # Map local tracker indices back to metadata animal IDs for genotype/line joins.
    if animal_lookup is not None and row.get("set", row_idx) in animal_lookup:
        values = np.array(animal_lookup[row.get("set", row_idx)]).astype(int)
    elif "anIDAll" in row and pd.notna(row["anIDAll"]):
        values = np.array(str(row["anIDAll"]).split()).astype(int)
    elif "anNr" in row and pd.notna(row["anNr"]):
        values = parse_an_nrs(row["anNr"])
    else:
        values = np.arange(n_animals)
    if len(values) < n_animals:
        raise ValueError("Animal ID list is shorter than inferred focal animal count.")
    return values[:n_animals]


def _episode_blocks(episode_all, episode_map, max_episode_blocks):
    # Identify contiguous runs first, then apply the 24-episode limit before filtering labels.
    labels = pd.Series(episode_all).map(normalize_episode_label)
    block_id = labels.ne(labels.shift()).cumsum() - 1
    rows = []
    for epi_nr, indices in pd.Series(np.arange(len(labels))).groupby(block_id):
        if max_episode_blocks is not None and max_episode_blocks > 0 and epi_nr >= max_episode_blocks:
            break
        start = int(indices.iloc[0])
        stop = int(indices.iloc[-1] + 1)
        episode = labels.iloc[start]
        if episode not in episode_map:
            continue
        prev_episode = labels.iloc[start - 1] if start > 0 else np.nan
        rows.append(
            {
                "epiNr": int(epi_nr),
                "episode": episode,
                "episode_type": episode_map[episode],
                "prev_episode": prev_episode,
                "prev_episode_type": episode_map.get(prev_episode, np.nan),
                "start": start,
                "stop": stop,
            }
        )
    return pd.DataFrame(rows)


def _is_escapee(meta):
    # Escapee animals are temporarily excluded by genotype labels such as esc_hi or esc_lo.
    genotype = str(meta.get("genotype", "")).strip().lower()
    return genotype.startswith("esc")


def _animal_frame_asd(asd_mm, blocks, row, row_idx, local_index, animal_id, meta, fps, frame_stride):
    # Expand one animal's ASD vector into a tidy table with absolute and within-episode time axes.
    parts = []
    for _, block in blocks.iterrows():
        frames = np.arange(int(block.start), int(block.stop), int(frame_stride))
        part = pd.DataFrame(
            {
                "experiment_index": row_idx,
                "experiment_key": _experiment_key(row, row_idx),
                "animalIndex_local": local_index,
                "animalIndex": animal_id,
                "frame_global": frames,
                "time_min": frames / (fps * 60),
                "episode_frame": frames - int(block.start),
                "episode_time_min": (frames - int(block.start)) / (fps * 60),
                "epiNr": int(block.epiNr),
                "episode": block.episode,
                "episode_type": block.episode_type,
                "prev_episode": block.prev_episode,
                "prev_episode_type": block.prev_episode_type,
                "asd_mm": asd_mm[frames],
            }
        )
        _attach_metadata(part, row, meta)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _animal_one_min_si(
    animal_xy,
    stim_xy,
    blocks,
    row,
    row_idx,
    local_index,
    animal_id,
    meta,
    fps,
    px_pmm,
    window_seconds,
    shuffle_min_shift_seconds,
    n_shift_runs,
):
    # Compute the short-window equivalent of Pair.ShoalIndex without changing Pair or experiment writers.
    rows = []
    window_frames = int(window_seconds * fps)
    min_shift_frames = int(shuffle_min_shift_seconds * fps)

    for _, block in blocks.iterrows():
        # Build observed ASD for the full original episode, then summarize it into 1-minute windows.
        start, stop = int(block.start), int(block.stop)
        animal_ep = animal_xy[start:stop]
        stim_ep = stim_xy[start:stop]
        asd_ep = np.linalg.norm(animal_ep - stim_ep, axis=1) / px_pmm
        n_windows = len(asd_ep) // window_frames
        shifts = _shift_values(len(asd_ep), min_shift_frames, n_shift_runs)

        # The null distribution rolls the stimulus trace within the episode by at least 30 seconds by default.
        shifted = []
        for shift in shifts:
            shifted_stim = np.roll(stim_ep, shift, axis=0)
            shifted.append(np.linalg.norm(animal_ep - shifted_stim, axis=1) / px_pmm)
        shifted = np.array(shifted)

        for minute in range(n_windows):
            # Match the existing SI sign convention: positive values mean observed ASD is smaller than shifted ASD.
            w0 = minute * window_frames
            w1 = w0 + window_frames
            observed = np.nanmean(asd_ep[w0:w1])
            shuffled = np.nanmean(shifted[:, w0:w1])
            si = (shuffled - observed) / shuffled if np.isfinite(shuffled) and shuffled != 0 else np.nan
            rows.append(
                {
                    "experiment_index": row_idx,
                    "experiment_key": _experiment_key(row, row_idx),
                    "animalIndex_local": local_index,
                    "animalIndex": animal_id,
                    "epiNr": int(block.epiNr),
                    "episode": block.episode,
                    "episode_type": block.episode_type,
                    "prev_episode": block.prev_episode,
                    "prev_episode_type": block.prev_episode_type,
                    "minute_in_episode": minute,
                    "minute_start_global": start + w0,
                    "minute_end_global": start + w1,
                    "time_min": (start + w0) / (fps * 60),
                    "observed_asd_mm": observed,
                    "shuffled_asd_mm": shuffled,
                    "si_1min": si,
                    "shuffle_min_shift_seconds": shuffle_min_shift_seconds,
                    "n_shift_runs": len(shifts),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        _attach_metadata(out, row, meta)
    return out


def _shift_values(length, min_shift_frames, n_shift_runs):
    # Use deterministic evenly spaced shifts so repeated notebook runs are reproducible.
    max_shift = length - min_shift_frames
    if max_shift <= min_shift_frames:
        return np.array([max(1, length // 2)], dtype=int)
    return np.unique(np.linspace(min_shift_frames, max_shift, n_shift_runs).astype(int))


def _attach_metadata(df, row, meta):
    # Add experiment and animal annotations needed for genotype-split plots and later filtering.
    for col in ["date", "date_label", "setup", "line", "set"]:
        if col in row:
            df[col] = row[col]
    for col in ["line", "genotype", "bg", "cohort", "generation", "repeat"]:
        if col in meta:
            df[col] = meta[col]
    if "genotype" in df:
        df["genotype_norm"] = df["genotype"].astype(str).str.strip().str.lower()
    if "line" in df and "date_label" in df:
        df["lineSet"] = df["line"].astype(str) + "_" + df["date_label"].astype(str)


def _experiment_key(row, row_idx):
    # Prefer the folder name for readable plot/debug labels; fall back to the raw file stem.
    if "folder" in row and pd.notna(row["folder"]):
        return str(row["folder"])
    return Path(str(row["txtPath"])).stem if "txtPath" in row else str(row_idx)


def _write_cache_settings(settings_path, cache_settings):
    if cache_settings is None:
        return
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(cache_settings), f, indent=2, sort_keys=True)


def _warn_if_cache_settings_mismatch(settings_path, cache_settings, row, row_idx):
    if cache_settings is None:
        return
    current = _json_ready(cache_settings)
    experiment_key = _experiment_key(row, row_idx)
    if not Path(settings_path).exists():
        _print_stale_cache_warning(
            experiment_key,
            "This cache has no settings metadata, so it cannot be checked against the current settings.",
            settings_path,
        )
        return

    with open(settings_path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    if cached == current:
        return

    changed = _changed_setting_names(cached, current)
    details = "Changed settings: " + ", ".join(changed) if changed else "Settings metadata changed."
    _print_stale_cache_warning(experiment_key, details, settings_path)


def _print_stale_cache_warning(experiment_key, details, settings_path):
    banner = "!" * 92
    print(banner, flush=True)
    print("WARNING: LOADING PROCESSED CARRYOVER DATA THAT MAY BE STALE", flush=True)
    print(f"Experiment: {experiment_key}", flush=True)
    print(details, flush=True)
    print(f"Cache settings file: {settings_path}", flush=True)
    print("Set FORCE_REPROCESS_RAW_DATA = True to rebuild this cache with the current settings.", flush=True)
    print(banner, flush=True)


def _changed_setting_names(cached, current):
    keys = sorted(set(cached) | set(current))
    return [key for key in keys if cached.get(key) != current.get(key)]


def _json_ready(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def _safe_filename(value):
    # Keep cache names readable while removing path separators and shell-unfriendly characters.
    text = str(value).strip()
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
