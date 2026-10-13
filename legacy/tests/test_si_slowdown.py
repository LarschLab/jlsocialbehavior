import unittest

import numpy as np
import pandas as pd

from legacy.functions.si_slowdown import _first_persistent, estimate_si_slowdown


class SISlowdownTests(unittest.TestCase):
    def test_persistent_crossing_ignores_transient_dip(self):
        self.assertEqual(_first_persistent([0.04, 0.07, 0.03], 0.05), 2)
        self.assertIsNone(_first_persistent([0.04, 0.07, 0.06], 0.05))

    def test_fraction_validation(self):
        with self.assertRaisesRegex(ValueError, "fraction"):
            estimate_si_slowdown(pd.DataFrame(), fraction=0)

    def test_mixed_model_and_cluster_bootstrap_smoke(self):
        rng = np.random.default_rng(8)
        rows = []
        for stimulus, offset in (("linear", 0), ("bout", 5)):
            for experiment in range(4):
                experiment_effect = rng.normal(0, 0.01)
                for fish in range(4):
                    fish_effect = rng.normal(0, 0.015)
                    for exposure, time in enumerate(offset + np.arange(8) * 10):
                        mean_si = 0.05 + 0.35 * (1 - np.exp(-time / 20))
                        rows.append(
                            {
                                "si": mean_si
                                + experiment_effect
                                + fish_effect
                                + rng.normal(0, 0.02),
                                "recording_min": time,
                                "episode_number": 2 * exposure + 1,
                                "episode_type": stimulus,
                                "animalSet": experiment,
                                "animalIndex": fish,
                                "genotype_norm": "hi" if fish < 2 else "lo",
                            }
                        )

        analysis = estimate_si_slowdown(
            pd.DataFrame(rows),
            spline_df=3,
            n_boot=2,
            min_bootstrap_valid=0.5,
            sensitivity_spline_dfs=(),
            run_leave_one_experiment_out=False,
            run_asymptotic_sensitivity=False,
            progress_every=None,
            random_state=3,
        )

        self.assertEqual(set(analysis.results["stimulus_type"]), {"linear", "bout"})
        self.assertTrue((analysis.results["total_fitted_increase"] > 0).all())
        self.assertTrue((analysis.results["bootstrap_valid_rate"] > 0).all())
        self.assertEqual(analysis.curve.groupby("stimulus_type").size().to_dict(), {"bout": 7, "linear": 7})


if __name__ == "__main__":
    unittest.main()
