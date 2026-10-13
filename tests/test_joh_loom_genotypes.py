import unittest
from unittest import mock

import numpy as np
import pandas as pd

from functions import joh_loom_helpers as loom


def synthetic_animals(n_frames=250, episode="CLfull020R"):
    frame = np.arange(n_frames, dtype=float)
    slopes = [1.0, 3.0, 5.0, 7.0]
    data = {}
    for animal, slope in enumerate(slopes, start=1):
        data[f"animal_{animal:02d}_x"] = frame * slope
        data[f"animal_{animal:02d}_y"] = np.zeros(n_frames)
        data[f"animal_{animal:02d}_orientation"] = np.zeros(n_frames)
    data["stim_x_embedded"] = np.zeros(n_frames)
    data["stim_y_embedded"] = np.zeros(n_frames)
    data["stim_size_embedded"] = np.full(n_frames, 20.0)
    data["episode_embedded"] = np.full(n_frames, episode, dtype=object)
    return pd.DataFrame(data)


class LoomGenotypeTests(unittest.TestCase):
    def test_full_and_semi_loom_episodes_parse_equivalently(self):
        stimulus = pd.DataFrame(
            {
                "episode": ["CLfull020R"] * 3 + ["CLsemi020R"] * 3,
                "stim_x": np.zeros(6),
                "stim_y": np.zeros(6),
                "stim_size": [0, 10, 20] * 2,
                "frame": np.arange(6),
                "block": [0] * 3 + [1] * 3,
            }
        )

        trials = loom.loom_trials(stimulus)

        self.assertEqual(trials[["episode", "loom_max_size", "side"]].to_dict("records"), [
            {"episode": "CLfull020R", "loom_max_size": 20, "side": "R"},
            {"episode": "CLsemi020R", "loom_max_size": 20, "side": "R"},
        ])

    def test_unsided_loom_block_is_split_at_each_positive_size_pulse(self):
        stimulus = pd.DataFrame(
            {
                "episode": ["CLsemi030"] * 500,
                "stim_x": np.zeros(500),
                "stim_y": np.zeros(500),
                "stim_size": np.zeros(500),
                "frame": np.arange(500),
                "block": np.zeros(500, dtype=int),
            }
        )
        stimulus.loc[150:170, "stim_size"] = np.linspace(30 / 21, 30, 21)
        stimulus.loc[350:370, "stim_size"] = np.linspace(30 / 21, 30, 21)

        trials = loom.unsided_loom_trials(stimulus, onset_in_block=150)

        self.assertEqual(trials["loom_onset_frame"].tolist(), [150, 350])
        self.assertEqual(trials["block_start_frame"].tolist(), [0, 200])
        self.assertEqual(trials["pulse_in_block"].tolist(), [1, 2])
        self.assertEqual(set(trials["episode"]), {"CLsemi030"})
        self.assertEqual(set(trials["side"]), {""})
        self.assertEqual(set(trials["loom_max_size"]), {30})

    def test_normalization_catalog_and_resolution(self):
        self.assertEqual(loom.normalize_genotype(" F2 "), "f2")
        self.assertIsNone(loom.normalize_genotype("na"))
        self.assertIsNone(loom.normalize_genotype(np.nan))

        animals = pd.DataFrame({"genotype": [" hi ", "LO", "F2", "na", None, "esc_hi"]})
        with mock.patch.object(pd, "read_excel", return_value=animals):
            self.assertEqual(loom.loom_genotype_catalog("metadata.xlsx"), ["esc_hi", "f2", "hi", "lo"])

        active, missing = loom.resolve_requested_genotypes(["HI", "lo", "F2"], ["hi", "f2"])
        self.assertEqual(active, ["hi", "f2"])
        self.assertEqual(missing, ["lo"])
        self.assertEqual(loom.resolve_requested_genotypes(["lo"], ["hi"]), ([], ["lo"]))

    def test_selection_attaches_normalized_animal_genotypes(self):
        experiments = pd.DataFrame(
            {"folder": ["exp"], "path": ["C:/data"], "anNr": ["1:4"], "date": ["01-01-2026"]}
        )
        animals = pd.DataFrame(
            {
                "anNr": [1, 2, 3, 4],
                "expDate": ["01-01-2026"] * 4,
                "line": ["A"] * 4,
                "genotype": [" hi ", "HI", "lo", "na"],
            }
        )

        def read_excel(_path, sheet_name):
            return experiments.copy() if sheet_name == "AllExp" else animals.copy()

        with mock.patch.object(pd, "read_excel", side_effect=read_excel), mock.patch(
            "functions.joh_loom_helpers.glob.glob", return_value=["C:/data/exp/PositionTxt.txt"]
        ):
            selected, missing = loom.select_loom_experiments("metadata.xlsx", include_genotypes=["LO"])

        self.assertTrue(missing.empty)
        self.assertEqual(selected.loc[0, "genotypes"], ["hi", "lo"])
        self.assertEqual(selected.loc[0, "animal_genotypes"], {1: "hi", 2: "hi", 3: "lo"})

    def test_selection_include_from_date_is_inclusive(self):
        experiments = pd.DataFrame(
            {
                "folder": ["before", "cutoff", "after", "missing"],
                "path": ["C:/data"] * 4,
                "anNr": ["1", "2", "3", "4"],
                "date": ["31-12-2025", "01-01-2026", "02-01-2026", "03-01-2026"],
            }
        )
        animals = pd.DataFrame(
            {
                "anNr": [1, 2, 3, 4],
                "expDate": ["31-12-2025", "01-01-2026", "02-01-2026", None],
                "line": ["A"] * 4,
                "genotype": ["hi"] * 4,
            }
        )

        def read_excel(_path, sheet_name):
            return experiments.copy() if sheet_name == "AllExp" else animals.copy()

        with mock.patch.object(pd, "read_excel", side_effect=read_excel), mock.patch(
            "functions.joh_loom_helpers.glob.glob",
            side_effect=lambda pattern: [f"C:/data/{pattern.split('/')[-2]}/PositionTxt.txt"],
        ):
            selected, _ = loom.select_loom_experiments(
                "metadata.xlsx", include_from_date="2026-01-01"
            )

        self.assertEqual(selected["folder"].tolist(), ["cutoff", "after"])

    def test_selection_exclude_date_range_is_inclusive(self):
        experiments = pd.DataFrame(
            {
                "folder": ["before", "start", "middle", "end", "after"],
                "path": ["C:/data"] * 5,
                "anNr": ["1", "2", "3", "4", "5"],
                "date": ["31-12-2025", "01-01-2026", "15-01-2026", "31-01-2026", "01-02-2026"],
            }
        )
        animals = pd.DataFrame(
            {
                "anNr": [1, 2, 3, 4, 5],
                "expDate": ["31-12-2025", "01-01-2026", "15-01-2026", "31-01-2026", "01-02-2026"],
                "line": ["A"] * 5,
                "genotype": ["hi"] * 5,
            }
        )

        def read_excel(_path, sheet_name):
            return experiments.copy() if sheet_name == "AllExp" else animals.copy()

        with mock.patch.object(pd, "read_excel", side_effect=read_excel), mock.patch(
            "functions.joh_loom_helpers.glob.glob",
            side_effect=lambda pattern: [f"C:/data/{pattern.split('/')[-2]}/PositionTxt.txt"],
        ):
            selected, _ = loom.select_loom_experiments(
                "metadata.xlsx",
                date_filter_mode="exclude_date_range",
                exclude_date_start="2026-01-01",
                exclude_date_end="2026-01-31",
            )

        self.assertEqual(selected["folder"].tolist(), ["before", "after"])

    def test_date_filter_validation(self):
        with self.assertRaisesRegex(ValueError, "date_filter_mode"):
            loom._validate_date_filter("unknown", "2026-01-01", None, None)
        with self.assertRaisesRegex(ValueError, "exclude_date_start"):
            loom._validate_date_filter("exclude_date_range", None, "2026-02-01", "2026-01-01")
        with self.assertRaisesRegex(ValueError, "include_from_date"):
            loom._validate_date_filter("include_from_date", "not-a-date", None, None)

    def test_default_velocity_summary_remains_pooled(self):
        animals = synthetic_animals()
        trials = loom.loom_trials(loom.embedded_stimulus_from_animal_file(animals))
        result = loom.extract_trial_velocity_summary(
            animals, trials, frame_end=210, step_frames=30, n_animals=4
        )
        self.assertNotIn("genotype", result)
        self.assertTrue(np.allclose(result["linear_velocity_mm_s"], 30.0))
        self.assertTrue(result["n_fish"].eq(4).all())

    def test_processed_tables_contain_exact_pooled_and_genotype_rows(self):
        animals = synthetic_animals()
        row = pd.Series(
            {
                "experiment": "synthetic",
                "experiment_index": 0,
                "txt_path": "synthetic.txt",
                "animal_ids": [101, 102, 103, 104],
                "animal_genotypes": {101: "HI", 102: "hi", 103: "lo", 104: "LO"},
            }
        )
        with mock.patch.object(loom, "load_animal_file", return_value=animals):
            tables = loom._process_loom_experiment(
                row,
                onset_in_block=150,
                pre_frames=50,
                post_frames=99,
                baseline_frames=(100, 140),
                response_frames=(149, 180),
                velocity_frame_start=0,
                velocity_frame_end=210,
                velocity_step_frames=30,
                fps=30,
                units_per_mm=4.0,
                max_velocity_window_frames=60,
            )

        sided_table_names = [
            "center_traces",
            "response_metrics",
            "trial_velocity",
            "trial_max_velocity",
            "fish_trial_max_velocity",
        ]
        for table_name in sided_table_names:
            table = tables[table_name]
            expected_genotypes = {"hi", "lo"} if table_name == "fish_trial_max_velocity" else {
                loom.POOLED_GENOTYPE,
                "hi",
                "lo",
            }
            self.assertEqual(set(table["genotype"]), expected_genotypes)
        self.assertTrue(tables["unsided_response_metrics"].empty)
        self.assertTrue(tables["unsided_fish_trial_max_velocity"].empty)

        velocity = tables["trial_velocity"]
        at_frame = velocity.loc[velocity["epFrame"].eq(30)].set_index("genotype")
        self.assertAlmostEqual(at_frame.loc[loom.POOLED_GENOTYPE, "linear_velocity_mm_s"], 30.0)
        self.assertAlmostEqual(at_frame.loc["hi", "linear_velocity_mm_s"], 15.0)
        self.assertAlmostEqual(at_frame.loc["lo", "linear_velocity_mm_s"], 45.0)
        self.assertEqual(at_frame.loc[loom.POOLED_GENOTYPE, "n_fish"], 4)
        self.assertEqual(at_frame.loc["hi", "n_fish"], 2)

        maxima = tables["trial_max_velocity"].set_index("genotype")
        self.assertAlmostEqual(maxima.loc[loom.POOLED_GENOTYPE, "trial_median_max_linear_velocity_mm_s"], 30.0)
        self.assertAlmostEqual(maxima.loc["hi", "trial_median_max_linear_velocity_mm_s"], 15.0)
        self.assertAlmostEqual(maxima.loc["lo", "trial_median_max_linear_velocity_mm_s"], 45.0)

        center = tables["center_traces"]
        center_counts = center.loc[center["relative_frame"].eq(0)].set_index("genotype")["n_animals"]
        self.assertEqual(center_counts.loc[loom.POOLED_GENOTYPE], 4)
        self.assertEqual(center_counts.loc["hi"], 2)
        self.assertEqual(center_counts.loc["lo"], 2)

        response = tables["response_metrics"]
        response_counts = response.loc[response["aggregation"].eq("side_resolved")].set_index("genotype")["n_animals"]
        self.assertEqual(response_counts.loc[loom.POOLED_GENOTYPE], 4)
        self.assertEqual(response_counts.loc["hi"], 2)
        self.assertEqual(response_counts.loc["lo"], 2)

        animal_trials = response.loc[response["aggregation"].eq("animal_trial")]
        self.assertEqual(len(animal_trials), 4)
        self.assertEqual(set(animal_trials["side"]), {"R"})
        self.assertEqual(set(animal_trials["animal_id"]), {101, 102, 103, 104})
        self.assertEqual(set(animal_trials["genotype"]), {"hi", "lo"})
        self.assertNotIn(loom.POOLED_GENOTYPE, set(animal_trials["genotype"]))

        fish_maxima = tables["fish_trial_max_velocity"].set_index("animal_id")
        self.assertEqual(set(fish_maxima["side"]), {"R"})
        self.assertEqual(set(fish_maxima["genotype"]), {"hi", "lo"})
        self.assertAlmostEqual(
            fish_maxima.loc[101, "fish_mean_pre_loom_linear_velocity_mm_s"], 7.5
        )
        self.assertAlmostEqual(
            fish_maxima.loc[102, "fish_mean_pre_loom_linear_velocity_mm_s"], 22.5
        )
        self.assertAlmostEqual(
            fish_maxima.loc[103, "fish_mean_pre_loom_linear_velocity_mm_s"], 37.5
        )
        self.assertAlmostEqual(
            fish_maxima.loc[104, "fish_mean_pre_loom_linear_velocity_mm_s"], 52.5
        )
        self.assertAlmostEqual(fish_maxima.loc[101, "fish_max_linear_velocity_mm_s"], 7.5)
        self.assertAlmostEqual(fish_maxima.loc[102, "fish_max_linear_velocity_mm_s"], 22.5)
        self.assertAlmostEqual(fish_maxima.loc[103, "fish_max_linear_velocity_mm_s"], 37.5)
        self.assertAlmostEqual(fish_maxima.loc[104, "fish_max_linear_velocity_mm_s"], 52.5)

    def test_processed_tables_accept_clsemi_only_recordings(self):
        animals = synthetic_animals(episode="CLsemi020R")
        row = pd.Series(
            {
                "experiment": "synthetic",
                "experiment_index": 0,
                "txt_path": "synthetic.txt",
                "animal_ids": [101, 102, 103, 104],
                "animal_genotypes": {101: "HI", 102: "hi", 103: "lo", 104: "LO"},
            }
        )

        with mock.patch.object(loom, "load_animal_file", return_value=animals):
            tables = loom._process_loom_experiment(
                row,
                onset_in_block=150,
                pre_frames=50,
                post_frames=99,
                baseline_frames=(100, 140),
                response_frames=(149, 180),
                velocity_frame_start=0,
                velocity_frame_end=210,
                velocity_step_frames=30,
                fps=30,
                units_per_mm=4.0,
                max_velocity_window_frames=60,
            )

        sided_table_names = [
            "center_traces",
            "response_metrics",
            "trial_velocity",
            "trial_max_velocity",
            "fish_trial_max_velocity",
        ]
        self.assertTrue(all(not tables[name].empty for name in sided_table_names))
        self.assertTrue(tables["unsided_response_metrics"].empty)
        self.assertTrue(tables["unsided_fish_trial_max_velocity"].empty)
        self.assertEqual(set(tables["trial_max_velocity"]["episode"]), {"CLsemi020R"})

    def test_unsided_pulses_are_separate_from_sided_analysis_tables(self):
        animals = synthetic_animals(n_frames=1000, episode="CLsemi030")
        animals["stim_size_embedded"] = 0.0
        animals.loc[150:170, "stim_size_embedded"] = np.linspace(30 / 21, 30, 21)
        animals.loc[350:370, "stim_size_embedded"] = np.linspace(30 / 21, 30, 21)
        animals.loc[500:, "episode_embedded"] = "CLsemi020R"
        animals.loc[500:, "stim_x_embedded"] = 20.0
        animals.loc[500:, "stim_size_embedded"] = 20.0
        row = pd.Series(
            {
                "experiment": "synthetic",
                "experiment_index": 0,
                "txt_path": "synthetic.txt",
                "animal_ids": [101, 102, 103, 104],
                "animal_genotypes": {101: "hi", 102: "hi", 103: "lo", 104: "lo"},
            }
        )

        with mock.patch.object(loom, "load_animal_file", return_value=animals):
            tables = loom._process_loom_experiment(
                row,
                onset_in_block=150,
                pre_frames=50,
                post_frames=99,
                baseline_frames=(100, 140),
                response_frames=(149, 180),
                velocity_frame_start=0,
                velocity_frame_end=300,
                velocity_step_frames=30,
                fps=30,
                units_per_mm=4.0,
                max_velocity_window_frames=120,
            )

        sided_trials = tables["response_metrics"].loc[
            tables["response_metrics"]["aggregation"].eq("animal_trial")
        ]
        self.assertEqual(set(sided_trials["episode"]), {"CLsemi020R"})
        self.assertEqual(set(tables["fish_trial_max_velocity"]["episode"]), {"CLsemi020R"})

        unsided_response = tables["unsided_response_metrics"]
        unsided_maxima = tables["unsided_fish_trial_max_velocity"]
        self.assertEqual(len(unsided_response), 8)
        self.assertEqual(len(unsided_maxima), 8)
        self.assertEqual(set(unsided_response["episode"]), {"CLsemi030"})
        self.assertEqual(set(unsided_maxima["episode"]), {"CLsemi030"})
        self.assertEqual(set(unsided_response["condition_trial"]), {1, 2})
        self.assertEqual(set(unsided_maxima["condition_trial"]), {1, 2})
        self.assertIn("fish_mean_pre_loom_linear_velocity_mm_s", unsided_maxima)
        self.assertTrue(
            np.allclose(
                unsided_maxima.loc[
                    unsided_maxima["animal_id"].eq(101),
                    "fish_mean_pre_loom_linear_velocity_mm_s",
                ],
                7.5,
            )
        )

    def test_pre_velocity_window_must_be_a_positive_step_multiple(self):
        animals = synthetic_animals()
        trials = loom.add_condition_trial_index(
            loom.loom_trials(loom.embedded_stimulus_from_animal_file(animals))
        )

        for invalid_window in (0, 31):
            with self.subTest(pre_window_frames=invalid_window), self.assertRaisesRegex(
                ValueError, "pre_window_frames"
            ):
                loom._trial_max_velocity_tables(
                    animals,
                    trials,
                    pre_window_frames=invalid_window,
                    window_frames=60,
                    step_frames=30,
                    n_animals=4,
                )

    def test_animal_trial_response_rows_keep_loom_sides_separate(self):
        animals = synthetic_animals(n_frames=500)
        animals.loc[:249, "episode_embedded"] = "CLfull020L"
        animals.loc[250:, "episode_embedded"] = "CLfull020R"
        row = pd.Series(
            {
                "experiment": "synthetic",
                "experiment_index": 0,
                "txt_path": "synthetic.txt",
                "animal_ids": [101, 102, 103, 104],
                "animal_genotypes": {101: "hi", 102: "hi", 103: "lo", 104: "lo"},
            }
        )

        with mock.patch.object(loom, "load_animal_file", return_value=animals):
            tables = loom._process_loom_experiment(
                row,
                onset_in_block=150,
                pre_frames=50,
                post_frames=99,
                baseline_frames=(100, 140),
                response_frames=(149, 180),
                velocity_frame_start=0,
                velocity_frame_end=210,
                velocity_step_frames=30,
                fps=30,
                units_per_mm=4.0,
                max_velocity_window_frames=60,
            )

        animal_trials = tables["response_metrics"].loc[
            tables["response_metrics"]["aggregation"].eq("animal_trial")
        ]
        self.assertEqual(len(animal_trials), 8)
        self.assertEqual(set(animal_trials["side"]), {"L", "R"})
        self.assertTrue(animal_trials.groupby(["animal_id", "side"]).size().eq(1).all())
        self.assertTrue(animal_trials["n_animals"].eq(1).all())

        fish_maxima = tables["fish_trial_max_velocity"]
        self.assertEqual(len(fish_maxima), 8)
        self.assertEqual(set(fish_maxima["side"]), {"L", "R"})
        self.assertTrue(fish_maxima.groupby(["animal_id", "side"]).size().eq(1).all())


if __name__ == "__main__":
    unittest.main()
