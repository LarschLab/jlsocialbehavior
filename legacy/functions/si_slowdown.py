"""Infer when further shoaling-index (SI) increases become practically small.

The analysis asks a forward-looking question at every observed exposure time:
how much does the fitted SI change before the *next exposure to the same
stimulus*, relative to the fitted SI increase across the complete analysis
window?  A slowdown is persistent only when that fraction is below the chosen
threshold at the candidate exposure and at every later exposure.

Two crossings are reported.  ``fitted_slowdown_*`` is the descriptive crossing
of the fitted curve.  ``supported_slowdown_*`` is deliberately more
conservative: the simultaneous one-sided bootstrap upper band must remain
below the threshold.  See :func:`estimate_si_slowdown` for the model hierarchy
and the exact interpretation of both crossings.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


@dataclass
class SISlowdownAnalysis:
    """Tables produced by :func:`estimate_si_slowdown`.

    ``results`` contains one summary row per stimulus. ``curve`` contains one
    row per forward same-stimulus interval and is the source for the two-panel
    slowdown figure. The remaining tables expose bootstrap draws, residual and
    coverage diagnostics, and model-sensitivity checks rather than hiding
    those checks inside the headline crossing estimate.
    """

    results: pd.DataFrame
    curve: pd.DataFrame
    bootstrap_crossings: pd.DataFrame
    diagnostics: pd.DataFrame
    coverage: pd.DataFrame
    spline_sensitivity: pd.DataFrame
    leave_one_experiment_out: pd.DataFrame


def _validate_settings(fraction, confidence, spline_df, n_boot):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if spline_df < 3:
        raise ValueError("spline_df must be at least 3")
    if n_boot < 0:
        raise ValueError("n_boot must be non-negative")


def _prepare_data(
    data,
    value_col,
    time_col,
    episode_col,
    stimulus_col,
    experiment_col,
    fish_col,
    genotype_col,
):
    required = [
        value_col,
        time_col,
        episode_col,
        stimulus_col,
        experiment_col,
        fish_col,
        genotype_col,
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = data.loc[:, required].copy()
    out = out.rename(
        columns={
            value_col: "_si",
            time_col: "_time",
            episode_col: "_episode",
            stimulus_col: "_stimulus",
            experiment_col: "_experiment",
            fish_col: "_fish",
            genotype_col: "_genotype",
        }
    )
    out["_si"] = pd.to_numeric(out["_si"], errors="coerce")
    out["_time"] = pd.to_numeric(out["_time"], errors="coerce")
    out["_episode"] = pd.to_numeric(out["_episode"], errors="coerce")
    out = out.dropna(
        subset=["_si", "_time", "_episode", "_stimulus", "_experiment", "_fish", "_genotype"]
    )
    if out.empty:
        raise ValueError("No complete SI observations remain after filtering")

    out["_stimulus"] = out["_stimulus"].astype(str)
    out["_experiment"] = out["_experiment"].astype(str)
    out["_fish"] = out["_fish"].astype(str)
    out["_genotype"] = out["_genotype"].astype(str)
    out["_fish_id"] = out["_experiment"] + "::" + out["_fish"]
    genotype_counts = out.groupby("_fish_id")["_genotype"].nunique()
    if (genotype_counts > 1).any():
        raise ValueError("Each fish must have exactly one genotype label")
    return out.sort_values(["_stimulus", "_experiment", "_fish_id", "_time"])


def _fit_mixed_model(data, time_term):
    # Fixed effects describe the cohort-average time curve while controlling
    # for genotype. Random experiment intercepts and fish-within-experiment
    # variance components retain the repeated-measures sampling hierarchy.
    formula = f"_si ~ {time_term} + C(_genotype)"
    model = smf.mixedlm(
        formula,
        data,
        groups=data["_experiment"],
        re_formula="1",
        vc_formula={"fish": "0 + C(_fish_id)"},
    )
    last_error = None
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                result = model.fit(reml=False, method=method, disp=False)
            if np.all(np.isfinite(result.fe_params)):
                return result
        except (np.linalg.LinAlgError, ValueError) as error:
            last_error = error
    raise RuntimeError("Mixed model did not produce finite fixed effects") from last_error


def _marginal_curve(result, data, times, time_transform=None):
    # A model prediction depends on the genotype reference coding. Average
    # predictions over the observed fish-level genotype proportions so the
    # reported curve represents the analyzed cohort rather than an arbitrary
    # reference genotype. Random effects are not added to these predictions.
    genotype_weights = (
        data[["_fish_id", "_genotype"]]
        .drop_duplicates()["_genotype"]
        .value_counts(normalize=True, sort=False)
    )
    prediction_rows = []
    for genotype, weight in genotype_weights.items():
        frame = pd.DataFrame(
            {
                "_time": times,
                "_genotype": genotype,
                "_experiment": data["_experiment"].iloc[0],
                "_fish_id": data["_fish_id"].iloc[0],
            }
        )
        if time_transform is not None:
            frame["_asymptotic_time"] = time_transform(times)
        prediction_rows.append(weight * np.asarray(result.predict(frame), dtype=float))
    return np.sum(prediction_rows, axis=0)


def _first_persistent(values, threshold):
    """Return the first index below ``threshold`` with no later reversal.

    A single early dip is not called a slowdown if a later increment returns
    to or above the threshold. Strict ``<`` implements the one-sided practical-
    equivalence criterion; equality is not treated as evidence of slowdown.
    """
    values = np.asarray(values, dtype=float)
    for index in range(values.size):
        remaining = values[index:]
        if np.all(np.isfinite(remaining)) and np.all(remaining < threshold):
            return index
    return None


def _curve_metrics(times, episodes, predicted, fraction):
    """Convert fitted SI levels into forward same-stimulus increments.

    Row ``i`` describes the interval ``times[i] -> times[i + 1]``. Therefore a
    crossing plotted at (for example) 100 minutes refers to the fitted change
    from 100 minutes to the next same-stimulus exposure, not to the SI level at
    100 minutes alone.
    """
    total_increase = float(predicted[-1] - predicted[0])
    increments = np.diff(predicted)
    ratios = np.full(increments.shape, np.nan, dtype=float)
    if total_increase > 0:
        ratios = increments / total_increase
    metrics = pd.DataFrame(
        {
            "recording_min": times[:-1],
            "next_recording_min": times[1:],
            "episode_number": episodes[:-1],
            "next_episode_number": episodes[1:],
            "fitted_si": predicted[:-1],
            "next_fitted_si": predicted[1:],
            "increment": increments,
            "increment_fraction": ratios,
        }
    )
    crossing_index = _first_persistent(ratios, fraction)
    return total_increase, metrics, crossing_index


def _fit_spline_curve(data, times, spline_df):
    result = _fit_mixed_model(
        data,
        f"cr(_time, df={int(spline_df)}, constraints='center')",
    )
    predicted = _marginal_curve(result, data, times)
    try:
        residual_values = np.asarray(result.resid)
    except (ValueError, np.linalg.LinAlgError):
        # A variance component can land exactly on zero while fixed-effect
        # predictions remain valid; use marginal residuals for diagnostics.
        residual_values = data["_si"].to_numpy() - np.asarray(result.predict(data))
    residual = pd.Series(residual_values, index=data.index)
    return predicted, residual


def _fit_asymptotic_curve(data, times):
    # This is a sensitivity model, not the primary inference model. Profile the
    # exponential time constant because it enters nonlinearly, then estimate
    # the remaining fixed and random effects with the same hierarchy as above.
    time_origin = float(np.min(times))
    positive_steps = np.diff(np.unique(times))
    positive_steps = positive_steps[positive_steps > 0]
    lower_tau = float(np.min(positive_steps) / 4) if positive_steps.size else 0.25
    upper_tau = max(float(np.ptp(times) * 10), lower_tau * 4)

    def fit_at_tau(log_tau):
        tau = float(np.exp(log_tau))
        work = data.copy()
        work["_asymptotic_time"] = 1 - np.exp(-(work["_time"] - time_origin) / tau)
        return _fit_mixed_model(work, "_asymptotic_time")

    def objective(log_tau):
        try:
            return -float(fit_at_tau(log_tau).llf)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.inf

    optimized = minimize_scalar(
        objective,
        bounds=(np.log(lower_tau), np.log(upper_tau)),
        method="bounded",
        options={"xatol": 0.03, "maxiter": 40},
    )
    if not optimized.success or not np.isfinite(optimized.fun):
        raise RuntimeError("Asymptotic mixed model could not profile a finite time constant")
    tau = float(np.exp(optimized.x))
    result = fit_at_tau(optimized.x)
    transform = lambda values: 1 - np.exp(-(np.asarray(values) - time_origin) / tau)
    return _marginal_curve(result, data, times, transform), tau


def _cluster_bootstrap(data, rng):
    """Draw one nonparametric experiment-cluster bootstrap sample.

    If the source contains ``E`` experiments, draw ``E`` experiment IDs with
    replacement and copy every fish/episode row belonging to each selected ID.
    An original experiment may therefore be absent, selected once, or selected
    multiple times. Repeated selections are assigned distinct ``boot_*`` group
    IDs, and fish IDs are rebuilt within those groups, so the mixed model treats
    the copies as separate bootstrap clusters. No independent resampling of
    fish, time points, or residuals occurs.
    """
    experiments = data["_experiment"].drop_duplicates().to_numpy()
    sampled = rng.choice(experiments, size=len(experiments), replace=True)
    parts = []
    for draw, experiment in enumerate(sampled):
        part = data.loc[data["_experiment"] == experiment].copy()
        part["_experiment"] = f"boot_{draw}"
        part["_fish_id"] = part["_experiment"] + "::" + part["_fish"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _residual_lag1(residual, data):
    residual_data = data.loc[:, ["_fish_id", "_time"]].copy()
    residual_data["residual"] = residual.reindex(data.index).to_numpy()
    correlations = []
    for _, fish_data in residual_data.groupby("_fish_id", sort=False):
        values = fish_data.sort_values("_time")["residual"].to_numpy()
        if values.size >= 3 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
            correlations.append(np.corrcoef(values[:-1], values[1:])[0, 1])
    if not correlations:
        return np.nan, 0
    return float(np.nanmedian(correlations)), len(correlations)


def _crossing_values(metrics, index):
    if index is None:
        return np.nan, np.nan
    row = metrics.iloc[index]
    return float(row["episode_number"]), float(row["recording_min"])


def estimate_si_slowdown(
    data,
    *,
    fraction=0.05,
    confidence=0.95,
    spline_df=4,
    n_boot=500,
    random_state=20260721,
    min_bootstrap_valid=0.8,
    sensitivity_spline_dfs=(3, 5, 6),
    run_leave_one_experiment_out=True,
    run_asymptotic_sensitivity=True,
    progress_every=25,
    value_col="si",
    time_col="recording_min",
    episode_col="episode_number",
    stimulus_col="episode_type",
    experiment_col="animalSet",
    fish_col="animalIndex",
    genotype_col="genotype_norm",
):
    """Fit SI curves and estimate persistent practical-slowdown crossings.

    The primary model is fitted separately for each stimulus type to the raw
    fish-by-episode rows. It uses a natural cubic spline for time, genotype as a
    fixed effect, an experiment random intercept, and a fish-within-experiment
    variance component. The returned marginal curve is averaged over the
    cohort's fish-level genotype proportions.

    For fitted values ``m(t)`` at consecutive same-stimulus exposure times, the
    plotted statistic is::

        increment_fraction(t) = (m(next_t) - m(t)) / (m(last_t) - m(first_t))

    Thus ``fraction=0.05`` defines "practically small" as a next-exposure change
    below 5% of the total fitted increase over this analysis window. The fitted
    crossing is the first time this statistic and every later statistic are
    below 0.05. This persistence rule prevents a transient dip from being
    reported as the slowdown point.

    Inference runs ``n_boot`` draws separately for each stimulus. Each draw
    samples the original number of experiments with replacement, retains every
    fish and episode inside each selected experiment, refits the primary model,
    and evaluates it on the original exposure-time grid. Duplicate selections
    receive distinct experiment and fish IDs. Genotype weights are recalculated
    from each resampled cohort.

    Successfully fitted bootstrap totals, including nonpositive totals, form a
    one-sided lower bound for the first-to-last SI increase. Interval fractions
    are usable only for draws with a positive total and finite ratios. The
    fraction of requested draws meeting that condition must reach
    ``min_bootstrap_valid``, and the lower bound on total increase must exceed
    zero, before inferential crossing results are reported.

    The bootstrap is then used in two distinct ways. First, percentile limits
    of the *finite* bootstrap crossing times describe variation in the fitted
    crossing conditional on a crossing occurring; ``bootstrap_crossing_rate``
    reports how often that happened among all requested draws. Second, for
    simultaneous inference, draw ``b`` is reduced to
    ``max_j(ratio_boot[b, j] - ratio_point[j])``. The confidence quantile of
    those maxima is added to every point fraction, producing a one-sided band
    that controls the complete interval curve rather than separate pointwise
    bounds. The supported crossing is the first position at which this band and
    every later value remain below ``fraction``.

    Parameters
    ----------
    fraction : float
        Practical-equivalence threshold expressed as a fraction of the total
        fitted increase, not as raw SI units.
    confidence : float
        Confidence level for the one-sided simultaneous band and for bootstrap
        summaries of the fitted crossing.
    spline_df : int
        Degrees of freedom for the primary natural cubic spline time effect.
    n_boot : int
        Number of experiment-cluster bootstrap refits per stimulus. Zero keeps
        descriptive curve output but disables uncertainty-supported inference.
    min_bootstrap_valid : float
        Minimum proportion of requested bootstrap draws that must yield finite
        positive-total increment fractions before inference is accepted.
    sensitivity_spline_dfs : sequence of int
        Alternative spline degrees of freedom used only in the sensitivity
        table; they do not alter the primary reported crossing.
    run_leave_one_experiment_out : bool
        Whether to refit after omitting each experiment in turn.
    run_asymptotic_sensitivity : bool
        Whether to compare the descriptive crossing with an exponential-
        asymptote mixed model.

    Returns
    -------
    SISlowdownAnalysis
        Headline results, interval-level fitted curves and bounds, bootstrap
        crossing draws, and diagnostic/sensitivity tables.

    Notes
    -----
    ``recording_min`` in the interval-level ``curve`` table is the *start* of
    the forward comparison; ``next_recording_min`` is its endpoint. When two
    stimuli alternate in five-minute episodes, consecutive exposures to one
    stimulus are normally ten minutes apart.
    """
    _validate_settings(fraction, confidence, spline_df, n_boot)
    if not 0 < min_bootstrap_valid <= 1:
        raise ValueError("min_bootstrap_valid must be greater than 0 and at most 1")
    prepared = _prepare_data(
        data,
        value_col,
        time_col,
        episode_col,
        stimulus_col,
        experiment_col,
        fish_col,
        genotype_col,
    )
    rng = np.random.default_rng(random_state)

    result_rows = []
    curve_parts = []
    bootstrap_rows = []
    diagnostic_rows = []
    sensitivity_rows = []
    leave_one_out_rows = []
    coverage_parts = []

    # Stimuli get separate time curves and separate bootstrap inferences. This
    # avoids forcing bout-like and linear episodes to share a progression shape.
    for stimulus, stimulus_data in prepared.groupby("_stimulus", sort=False):
        # Collapse only the time/episode lookup here. The mixed model below still
        # receives every original fish observation, not an episode-level mean.
        time_episode = (
            stimulus_data.groupby("_time", as_index=False)["_episode"].median().sort_values("_time")
        )
        times = time_episode["_time"].to_numpy(dtype=float)
        episodes = time_episode["_episode"].to_numpy(dtype=float)
        if times.size < spline_df + 1:
            raise ValueError(
                f"Stimulus {stimulus!r} has {times.size} time points; "
                f"at least {spline_df + 1} are required"
            )

        # Primary descriptive curve and its first persistent threshold crossing.
        predicted, residual = _fit_spline_curve(stimulus_data, times, spline_df)
        total_increase, metrics, crossing_index = _curve_metrics(
            times, episodes, predicted, fraction
        )
        metrics.insert(0, "stimulus_type", stimulus)
        curve_parts.append(metrics)
        fitted_episode, fitted_time = _crossing_values(metrics, crossing_index)
        descriptive_fitted_episode = fitted_episode
        descriptive_fitted_time = fitted_time
        lag1, n_lag1_fish = _residual_lag1(residual, stimulus_data)
        diagnostic_rows.append(
            {
                "stimulus_type": stimulus,
                "median_fish_residual_lag1": lag1,
                "n_fish_with_lag1": n_lag1_fish,
            }
        )
        # Coverage makes late-window loss of fish or experiments visible when
        # interpreting a slowdown supported near the end of the recording.
        coverage = (
            stimulus_data.groupby(["_time", "_episode"], as_index=False)
            .agg(
                n_observations=("_si", "size"),
                n_fish=("_fish_id", "nunique"),
                n_experiments=("_experiment", "nunique"),
            )
            .rename(columns={"_time": "recording_min", "_episode": "episode_number"})
        )
        coverage.insert(0, "stimulus_type", stimulus)
        coverage_parts.append(coverage)

        # Store full interval trajectories, not just crossing times: simultaneous
        # inference below must control the upper band across every interval.
        boot_ratios = []
        boot_totals = []
        boot_crossing_times = []
        boot_crossing_episodes = []
        for bootstrap_index in range(n_boot):
            if progress_every and bootstrap_index % progress_every == 0:
                print(
                    f"SI slowdown bootstrap {stimulus}: "
                    f"{bootstrap_index}/{n_boot}",
                    end="\r",
                )
            # Draw the original number of experiment clusters with replacement;
            # all rows within a selected cluster travel together.
            bootstrap_data = _cluster_bootstrap(stimulus_data, rng)
            try:
                # Refit the same primary model and predict on the original time
                # grid so interval fractions align across all bootstrap draws.
                boot_predicted, _ = _fit_spline_curve(bootstrap_data, times, spline_df)
                boot_total, boot_metrics, boot_crossing = _curve_metrics(
                    times, episodes, boot_predicted, fraction
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                # Failed mixed-model fits remain in the requested-draw
                # denominator and therefore reduce bootstrap_valid_rate.
                continue
            # Keep every successfully fitted total for the positive-increase
            # gate, including zero or negative totals.
            boot_totals.append(boot_total)
            if boot_total > 0 and np.all(np.isfinite(boot_metrics["increment_fraction"])):
                # Only a positive total gives interpretable normalized
                # increments. A usable curve may still have no persistent
                # crossing, represented by NaN in the crossing columns.
                boot_ratios.append(boot_metrics["increment_fraction"].to_numpy())
                boot_episode, boot_time = _crossing_values(boot_metrics, boot_crossing)
                boot_crossing_episodes.append(boot_episode)
                boot_crossing_times.append(boot_time)
                bootstrap_rows.append(
                    {
                        "stimulus_type": stimulus,
                        "bootstrap_index": bootstrap_index,
                        "total_increase": boot_total,
                        "fitted_crossing_episode": boot_episode,
                        "fitted_crossing_min": boot_time,
                    }
                )
        if progress_every and n_boot:
            print(f"SI slowdown bootstrap {stimulus}: {n_boot}/{n_boot}")

        # Both rates use all requested draws as denominator. The crossing rate
        # can be smaller because a valid fraction curve need not cross.
        bootstrap_valid_rate = len(boot_ratios) / n_boot if n_boot else np.nan
        finite_boot_crossings = np.asarray(boot_crossing_times, dtype=float)
        finite_boot_crossings = finite_boot_crossings[np.isfinite(finite_boot_crossings)]
        bootstrap_crossing_rate = len(finite_boot_crossings) / n_boot if n_boot else np.nan
        # At 95% confidence this is the 5th percentile of all successfully
        # fitted totals. Do not interpret slowdown fractions unless this
        # one-sided lower bound supports a positive complete-window increase.
        total_lower = (
            float(np.quantile(boot_totals, 1 - confidence)) if boot_totals else np.nan
        )
        increase_supported = bool(
            total_increase > 0
            and n_boot > 0
            and bootstrap_valid_rate >= min_bootstrap_valid
            and np.isfinite(total_lower)
            and total_lower > 0
        )

        crossing_ci_low = np.nan
        crossing_ci_high = np.nan
        supported_index = None
        upper_band = np.full(len(metrics), np.nan)
        if increase_supported and finite_boot_crossings.size:
            # This interval describes variability in the *descriptive crossing
            # time*. It is conditional on a finite crossing and is distinct from
            # the supported crossing defined below; crossing_rate exposes how
            # many requested draws contributed finite times.
            alpha = 1 - confidence
            crossing_ci_low, crossing_ci_high = np.quantile(
                finite_boot_crossings, [alpha / 2, 1 - alpha / 2]
            )
        if increase_supported and boot_ratios:
            ratio_matrix = np.asarray(boot_ratios)
            point_ratios = metrics["increment_fraction"].to_numpy(dtype=float)
            # For each draw b, compute D_b = max_j(r*_bj - rhat_j). The requested
            # confidence quantile of D_b is one critical value shared by all
            # intervals: U_j = rhat_j + critical. Taking the maximum before the
            # quantile controls the band simultaneously rather than making a
            # separate pointwise claim at every exposure.
            max_positive_deviation = np.max(ratio_matrix - point_ratios, axis=1)
            simultaneous_critical = np.quantile(max_positive_deviation, confidence)
            upper_band = point_ratios + simultaneous_critical
            supported_index = _first_persistent(upper_band, fraction)

        metrics["simultaneous_upper_fraction"] = upper_band
        metrics["slowdown_fraction"] = fraction
        supported_episode, supported_time = _crossing_values(metrics, supported_index)
        if not increase_supported:
            fitted_episode = np.nan
            fitted_time = np.nan
            crossing_ci_low = np.nan
            crossing_ci_high = np.nan

        asymptotic_episode = np.nan
        asymptotic_time = np.nan
        asymptotic_tau = np.nan
        # A substantially different asymptotic-model crossing is flagged as
        # model dependence; it does not replace the primary spline estimate.
        if run_asymptotic_sensitivity:
            try:
                asymptotic_predicted, asymptotic_tau = _fit_asymptotic_curve(
                    stimulus_data, times
                )
                _, asymptotic_metrics, asymptotic_index = _curve_metrics(
                    times, episodes, asymptotic_predicted, fraction
                )
                asymptotic_episode, asymptotic_time = _crossing_values(
                    asymptotic_metrics, asymptotic_index
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass

        if total_increase <= 0:
            inference_status = "non_positive_fitted_total_increase"
        elif n_boot == 0:
            inference_status = "bootstrap_not_run"
        elif bootstrap_valid_rate < min_bootstrap_valid:
            inference_status = "insufficient_valid_bootstrap_samples"
        elif not np.isfinite(total_lower) or total_lower <= 0:
            inference_status = "positive_total_increase_not_supported"
        else:
            inference_status = "positive_total_increase_supported"

        result_rows.append(
            {
                "stimulus_type": stimulus,
                "fitted_slowdown_episode": fitted_episode,
                "fitted_slowdown_min": fitted_time,
                "fitted_slowdown_ci_low_min": crossing_ci_low,
                "fitted_slowdown_ci_high_min": crossing_ci_high,
                "supported_slowdown_episode": supported_episode,
                "supported_slowdown_min": supported_time,
                "total_fitted_increase": total_increase,
                "total_increase_one_sided_lower": total_lower,
                "threshold_increment": fraction * total_increase,
                "slowdown_fraction": fraction,
                "bootstrap_valid_rate": bootstrap_valid_rate,
                "bootstrap_crossing_rate": bootstrap_crossing_rate,
                "increase_supported": increase_supported,
                "inference_status": inference_status,
                "asymptotic_slowdown_episode": asymptotic_episode,
                "asymptotic_slowdown_min": asymptotic_time,
                "asymptotic_tau_min": asymptotic_tau,
                "model_dependent": bool(
                    np.isfinite(fitted_time)
                    and np.isfinite(asymptotic_time)
                    and fitted_time != asymptotic_time
                ),
            }
        )

        # Record the primary fit beside alternative spline complexities so the
        # descriptive crossing's sensitivity to curve flexibility is auditable.
        sensitivity_rows.append(
            {
                "stimulus_type": stimulus,
                "spline_df": spline_df,
                "total_fitted_increase": total_increase,
                "fitted_slowdown_episode": descriptive_fitted_episode,
                "fitted_slowdown_min": descriptive_fitted_time,
                "is_primary_model": True,
            }
        )
        for candidate_df in sensitivity_spline_dfs:
            if candidate_df == spline_df or times.size < candidate_df + 1:
                continue
            try:
                candidate_predicted, _ = _fit_spline_curve(
                    stimulus_data, times, candidate_df
                )
                candidate_total, candidate_metrics, candidate_index = _curve_metrics(
                    times, episodes, candidate_predicted, fraction
                )
                candidate_episode, candidate_time = _crossing_values(
                    candidate_metrics, candidate_index
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                candidate_total = candidate_episode = candidate_time = np.nan
            sensitivity_rows.append(
                {
                    "stimulus_type": stimulus,
                    "spline_df": candidate_df,
                    "total_fitted_increase": candidate_total,
                    "fitted_slowdown_episode": candidate_episode,
                    "fitted_slowdown_min": candidate_time,
                    "is_primary_model": False,
                }
            )

        # These refits reveal whether one experiment determines the descriptive
        # slowdown. They intentionally do not repeat the expensive bootstrap.
        if run_leave_one_experiment_out:
            for omitted in stimulus_data["_experiment"].drop_duplicates():
                reduced = stimulus_data.loc[stimulus_data["_experiment"] != omitted]
                try:
                    reduced_predicted, _ = _fit_spline_curve(reduced, times, spline_df)
                    reduced_total, reduced_metrics, reduced_index = _curve_metrics(
                        times, episodes, reduced_predicted, fraction
                    )
                    reduced_episode, reduced_time = _crossing_values(
                        reduced_metrics, reduced_index
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    reduced_total = reduced_episode = reduced_time = np.nan
                leave_one_out_rows.append(
                    {
                        "stimulus_type": stimulus,
                        "omitted_experiment": omitted,
                        "total_fitted_increase": reduced_total,
                        "fitted_slowdown_episode": reduced_episode,
                        "fitted_slowdown_min": reduced_time,
                    }
                )

    return SISlowdownAnalysis(
        results=pd.DataFrame(result_rows),
        curve=pd.concat(curve_parts, ignore_index=True),
        bootstrap_crossings=pd.DataFrame(bootstrap_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
        coverage=pd.concat(coverage_parts, ignore_index=True),
        spline_sensitivity=pd.DataFrame(sensitivity_rows),
        leave_one_experiment_out=pd.DataFrame(leave_one_out_rows),
    )
