import unittest

import numpy as np

from functions.arena_use import calculate_arena_use_evenness


class ArenaUseEvennessTests(unittest.TestCase):
    def test_concentrated_and_uniform_occupancy(self):
        xy = np.full((4, 2, 2), -0.5, dtype=float)
        xy[:, 1, :] = np.array(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
        )

        result = calculate_arena_use_evenness(xy, 1.0, 2, 1.0)

        self.assertAlmostEqual(result.evenness[0], 0.0)
        self.assertAlmostEqual(result.evenness[1], 1.0)
        self.assertEqual(result.arena_bin_count, 4)
        self.assertTrue(np.allclose(result.bin_edges, [-1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(np.nansum(result.probability, axis=(1, 2)), 1.0))

    def test_restricted_fish_uses_full_arena_denominator(self):
        xy = np.array(
            [[[-0.5, -0.5]], [[-0.5, -0.5]], [[0.5, -0.5]], [[0.5, -0.5]]]
        )

        result = calculate_arena_use_evenness(xy, 1.0, 2, 1.0)

        self.assertAlmostEqual(result.evenness[0], 0.5)
        self.assertEqual(result.visited_bin_count[0], 2)
        self.assertAlmostEqual(result.coverage_fraction[0], 0.5)

    def test_out_of_arena_samples_do_not_change_distribution(self):
        baseline = np.array([[[-0.5, -0.5]], [[0.5, -0.5]]])
        with_outlier = np.concatenate([baseline, np.array([[[20.0, 20.0]]])])

        expected = calculate_arena_use_evenness(baseline, 1.0, 2, 1.0)
        observed = calculate_arena_use_evenness(with_outlier, 1.0, 2, 1.0)

        self.assertTrue(np.allclose(expected.probability, observed.probability, equal_nan=True))
        self.assertTrue(np.allclose(expected.evenness, observed.evenness, equal_nan=True))
        self.assertEqual(observed.valid_in_arena_samples[0], 2)

    def test_sample_interval_changes_occupancy_but_not_probability(self):
        xy = np.array([[[-0.5, -0.5]], [[0.5, -0.5]]])

        counts = calculate_arena_use_evenness(xy, 1.0, 2, 1.0)
        seconds = calculate_arena_use_evenness(xy, 1.0, 2, 2.5)

        self.assertTrue(
            np.allclose(seconds.occupancy_seconds, counts.occupancy_seconds * 2.5, equal_nan=True)
        )
        self.assertTrue(np.allclose(counts.probability, seconds.probability, equal_nan=True))
        self.assertTrue(np.allclose(counts.evenness, seconds.evenness, equal_nan=True))

    def test_empty_occupancy_is_unscorable(self):
        result = calculate_arena_use_evenness(
            np.full((3, 1, 2), np.nan),
            1.0,
            2,
            1.0,
        )

        self.assertTrue(np.isnan(result.evenness[0]))
        self.assertTrue(np.isnan(result.probability[0]).all())
        self.assertEqual(result.total_in_arena_seconds[0], 0.0)

    def test_input_validation(self):
        valid_xy = np.zeros((2, 1, 2))
        invalid_calls = [
            (np.zeros((2, 2)), 1.0, 2, 1.0),
            (valid_xy, 0.0, 2, 1.0),
            (valid_xy, 1.0, 1, 1.0),
            (valid_xy, 1.0, 2.5, 1.0),
            (valid_xy, 1.0, 2, 0.0),
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                calculate_arena_use_evenness(*arguments)


if __name__ == "__main__":
    unittest.main()
