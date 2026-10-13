import unittest

import numpy as np

from functions.dish_use import analyze_dish_use


class DishUseEvennessTests(unittest.TestCase):
    def test_concentrated_and_uniform_occupancy(self):
        xy = np.full((4, 2, 2), -0.5, dtype=float)
        xy[:, 1, :] = np.array(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
        )

        result = analyze_dish_use(xy, 1.0, 2)

        self.assertAlmostEqual(result.evenness[0], 0.0)
        self.assertAlmostEqual(result.evenness[1], 1.0)
        self.assertEqual(result.dish_bin_count, 4)
        self.assertTrue(np.allclose(result.bin_edges, [-1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(np.nansum(result.probability, axis=(1, 2)), 1.0))
        self.assertEqual(result.total_binned_frame_count.tolist(), [4, 4])

    def test_restricted_fish_uses_full_dish_denominator(self):
        xy = np.array(
            [[[-0.5, -0.5]], [[-0.5, -0.5]], [[0.5, -0.5]], [[0.5, -0.5]]]
        )

        result = analyze_dish_use(xy, 1.0, 2)

        self.assertAlmostEqual(result.evenness[0], 0.5)
        self.assertEqual(result.visited_bin_count[0], 2)
        self.assertAlmostEqual(result.coverage_fraction[0], 0.5)

    def test_out_of_dish_samples_do_not_change_distribution(self):
        baseline = np.array([[[-0.5, -0.5]], [[0.5, -0.5]]])
        with_outlier = np.concatenate([baseline, np.array([[[20.0, 20.0]]])])

        expected = analyze_dish_use(baseline, 1.0, 2)
        observed = analyze_dish_use(with_outlier, 1.0, 2)

        self.assertTrue(np.allclose(expected.probability, observed.probability, equal_nan=True))
        self.assertTrue(np.allclose(expected.evenness, observed.evenness, equal_nan=True))
        self.assertEqual(observed.valid_in_dish_samples[0], 2)
        self.assertEqual(observed.total_binned_frame_count[0], 2)

    def test_frame_counts_define_probability_and_evenness(self):
        xy = np.array([[[-0.5, -0.5]]] * 3 + [[[-0.5, 0.5]]])

        result = analyze_dish_use(xy, 1.0, 2)

        self.assertTrue(np.array_equal(result.occupancy_frame_counts[0], [[3, 0], [1, 0]]))
        self.assertEqual(result.total_binned_frame_count[0], 4)
        self.assertTrue(np.array_equal(result.probability[0], [[0.75, 0], [0.25, 0]]))
        expected_entropy = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
        self.assertAlmostEqual(result.evenness[0], expected_entropy / np.log(4))

    def test_empty_occupancy_is_unscorable(self):
        result = analyze_dish_use(
            np.full((3, 1, 2), np.nan),
            1.0,
            2,
        )

        self.assertTrue(np.isnan(result.evenness[0]))
        self.assertTrue(np.isnan(result.probability[0]).all())
        self.assertEqual(result.total_binned_frame_count[0], 0)

    def test_input_validation(self):
        valid_xy = np.zeros((2, 1, 2))
        invalid_calls = [
            (np.zeros((2, 2)), 1.0, 2),
            (valid_xy, 0.0, 2),
            (valid_xy, 1.0, 1),
            (valid_xy, 1.0, 2.5),
        ]
        for arguments in invalid_calls:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                analyze_dish_use(*arguments)


if __name__ == "__main__":
    unittest.main()
