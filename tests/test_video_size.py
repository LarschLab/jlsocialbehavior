import io
import logging
import os
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

# Match the notebook entrypoint's import order to avoid existing circular imports.
logging.basicConfig()
import functions.notebookHelper
from functions import getAnimalSizeFromVideo as size_video
from functions import getVideoProperties as video_properties
from functions import video_functions as vf


REPO_ROOT = Path(__file__).resolve().parents[1]


class VideoTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp_dir = self.stack.enter_context(
            tempfile.TemporaryDirectory(prefix='.video_test_', dir=REPO_ROOT)
        )
        self.stdout = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))

    def patch(self, target, **kwargs):
        return self.stack.enter_context(mock.patch(target, **kwargs))


class FrameExtractionTests(VideoTestCase):
    def setUp(self):
        super().setUp()
        self.frame_dir = os.path.join(self.temp_dir, 'frames')
        os.mkdir(self.frame_dir)
        self.patch('functions.video_functions.tempfile.mkdtemp', return_value=self.frame_dir)
        self.patch('functions.getVideoProperties.getVideoProperties',
                   return_value={'fps': 30, 'nb_frames': '100'})
        self.run = self.patch('functions.video_functions.subprocess.run')
        self.files = [os.path.join(self.frame_dir, f'frame_{i:04d}.png') for i in range(1, 4)]
        self.patch('functions.video_functions.glob.glob', return_value=self.files)

    def test_selects_sorted_unique_frames_with_video_length_filtering(self):
        temp_dir, files, frames = vf.extract_frames_ffmpeg(
            'video.avi', np.array([30, 10, 10, 20, 200])
        )

        np.testing.assert_array_equal(frames, [10, 20, 30])
        self.assertEqual(files, self.files)
        self.assertEqual(temp_dir, self.frame_dir)
        command = self.run.call_args.args[0]
        self.assertEqual(command[command.index('-vf') + 1],
                         r'select=eq(n\,10)+eq(n\,20)+eq(n\,30)')

    def test_no_frames_within_video_length_cleans_up(self):
        self.assertEqual(vf.extract_frames_ffmpeg('video.avi', np.array([100, 200])),
                         (None, None, None))
        self.run.assert_not_called()
        self.assertFalse(os.path.exists(self.frame_dir))

    def test_ffmpeg_failure_cleans_up(self):
        for error in (FileNotFoundError(),
                      subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'failed')):
            with self.subTest(error=type(error).__name__):
                if not os.path.exists(self.frame_dir):
                    os.mkdir(self.frame_dir)
                self.run.side_effect = error
                self.assertEqual(vf.extract_frames_ffmpeg('video.avi', np.array([10])),
                                 (None, None, None))
                self.assertFalse(os.path.exists(self.frame_dir))

    def test_duplicate_requests_keep_original_coordinate_rows(self):
        frames = np.array([30, 10, 10, 20, 200])
        coordinates = np.arange(10).reshape(5, 1, 2).astype(float)
        self.patch('functions.video_functions.cv2.imread',
                   side_effect=lambda path, *args: np.full(
                       (4, 4), (self.files.index(path) + 1) * 10, dtype=np.uint8))
        capture = self.patch('functions.video_functions.cv2.VideoCapture')
        crop_calls = []

        def crop(image, center, box_size):
            crop_calls.append((int(image[0, 0]), center.x, center.y))
            return image

        self.patch('functions.video_functions.ImageProcessor.crop_zero_pad', side_effect=crop)
        self.patch('functions.video_functions.ImageProcessor.to_binary',
                   side_effect=lambda image, *args, **kwargs: image)
        self.patch('functions.video_functions.cv2.findContours', return_value=([], None))
        self.patch('functions.video_functions.ImageProcessor.get_contour_containing_point',
                   return_value=[np.zeros((4, 1, 2), dtype=np.int32)])
        self.patch('functions.video_functions.cv2.minAreaRect',
                   return_value=((1, 2), (3, 4), 5))
        self.patch('functions.video_functions.cv2.drawContours')
        self.patch('functions.video_functions.cv2.mean',
                   side_effect=lambda image, **kwargs: (float(image[0, 0]), 0, 0, 0))

        result = vf.getAnimalLength('video.avi', frames, coordinates)

        self.assertEqual(crop_calls, [(30, 0, 1), (10, 2, 3), (10, 4, 5), (20, 6, 7)])
        np.testing.assert_array_equal(result[:4, 0, 5], [30, 10, 10, 20])
        self.assertTrue(np.isnan(result[4]).all())
        self.assertEqual(result.shape, (5, 1, 6))
        capture.assert_not_called()
        self.assertFalse(os.path.exists(self.frame_dir))


class AnimalLengthCleanupTests(VideoTestCase):
    def setUp(self):
        super().setUp()
        self.capture = self.patch('functions.video_functions.cv2.VideoCapture')
        self.capture.return_value.read.return_value = (False, None)
        self.frames = np.array([10, 20])
        self.coordinates = np.zeros((2, 1, 2))

    def test_disabled_or_failed_ffmpeg_opens_and_releases_capture(self):
        extraction = self.patch('functions.video_functions.extract_frames_ffmpeg',
                                return_value=(None, None, None))
        for use_ffmpeg in (False, True):
            with self.subTest(use_ffmpeg=use_ffmpeg):
                self.capture.reset_mock()
                extraction.reset_mock()
                result = vf.getAnimalLength('video.avi', self.frames, self.coordinates,
                                           use_ffmpeg=use_ffmpeg)
                self.assertTrue(np.isnan(result).all())
                self.capture.assert_called_once_with('video.avi')
                self.capture.return_value.release.assert_called_once_with()
                self.assertEqual(extraction.call_count, int(use_ffmpeg))
                self.assertEqual(self.capture.return_value.set.call_args_list,
                                 [mock.call(vf.cv2.CAP_PROP_POS_FRAMES, 10),
                                  mock.call(vf.cv2.CAP_PROP_POS_FRAMES, 20)])

    def test_incomplete_ffmpeg_output_is_cleaned_before_fallback(self):
        frame_dir = os.path.join(self.temp_dir, 'frames')
        os.mkdir(frame_dir)
        self.patch('functions.video_functions.extract_frames_ffmpeg',
                   return_value=(frame_dir, ['one.png'], self.frames))
        imread = self.patch('functions.video_functions.cv2.imread')

        def open_capture(path):
            self.assertFalse(os.path.exists(frame_dir))
            return self.capture.return_value

        self.capture.side_effect = open_capture
        result = vf.getAnimalLength('video.avi', self.frames, self.coordinates)

        self.assertTrue(np.isnan(result).all())
        imread.assert_not_called()
        self.capture.return_value.release.assert_called_once_with()

    def test_capture_is_released_after_processing_exception(self):
        self.capture.return_value.read.return_value = (True, np.zeros((4, 4), dtype=np.uint8))
        self.patch('functions.video_functions.ImageProcessor.crop_zero_pad',
                   side_effect=RuntimeError('processing failed'))

        with self.assertRaisesRegex(RuntimeError, 'processing failed'):
            vf.getAnimalLength('video.avi', self.frames, self.coordinates, use_ffmpeg=False)

        self.capture.return_value.release.assert_called_once_with()

    def test_ffmpeg_directory_is_removed_after_image_loading_exception(self):
        frame_dir = os.path.join(self.temp_dir, 'frames')
        os.mkdir(frame_dir)
        self.patch('functions.video_functions.extract_frames_ffmpeg',
                   return_value=(frame_dir, ['one.png', 'two.png'], self.frames))
        self.patch('functions.video_functions.cv2.imread', side_effect=RuntimeError('read failed'))

        with self.assertRaisesRegex(RuntimeError, 'read failed'):
            vf.getAnimalLength('video.avi', self.frames, self.coordinates)

        self.assertFalse(os.path.exists(frame_dir))
        self.capture.assert_not_called()

    def test_ffmpeg_directory_is_removed_after_processing_exception(self):
        frame_dir = os.path.join(self.temp_dir, 'frames')
        os.mkdir(frame_dir)
        self.patch('functions.video_functions.extract_frames_ffmpeg',
                   return_value=(frame_dir, ['one.png', 'two.png'], self.frames))
        self.patch('functions.video_functions.cv2.imread',
                   return_value=np.zeros((4, 4), dtype=np.uint8))
        self.patch('functions.video_functions.ImageProcessor.crop_zero_pad',
                   side_effect=RuntimeError('processing failed'))

        with self.assertRaisesRegex(RuntimeError, 'processing failed'):
            vf.getAnimalLength('video.avi', self.frames, self.coordinates)

        self.assertFalse(os.path.exists(frame_dir))
        self.capture.assert_not_called()


class AnimalSizeSamplingTests(VideoTestCase):
    def test_roi_samples_are_distinct_and_match_trajectory_rows(self):
        roi_path = os.path.join(self.temp_dir, 'ROI.csv')
        self.patch('functions.getVideoProperties.getVideoProperties',
                   return_value={'width': '512', 'height': '512'})
        self.patch('functions.notebookHelper.load_data_flexDelim',
                   return_value=np.array([[10, 20], [30, 40]]))
        frame_ids = np.arange(20000)
        raw_data = pd.DataFrame({
            'x0': frame_ids, 'y0': frame_ids * 2, 'angle0': 0,
            'x1': frame_ids * 3, 'y1': frame_ids * 4, 'angle1': 0,
        })

        def measure(path, frames, coordinates, **kwargs):
            self.assertEqual(len(frames), 400)
            self.assertTrue(np.all(np.diff(frames) > 0))
            self.assertTrue(np.all((frames >= 1000) & (frames < 20000)))
            np.testing.assert_array_equal(coordinates[:, 0, :],
                                          np.column_stack([frames + 10, frames * 2 + 20]))
            np.testing.assert_array_equal(coordinates[:, 1, :],
                                          np.column_stack([frames * 3 + 30, frames * 4 + 40]))
            result = np.zeros((400, 2, 6))
            result[:, :, 2:4] = [12, 8]
            return result

        measurement = self.patch('functions.video_functions.getAnimalLength', side_effect=measure)
        # This seed reproducibly generates duplicate frames with the old randint sampler.
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        np.random.seed(0)
        result = size_video.getAnimalSizeFromVideo(
            'video.avi', raw_data, camHeight=0, numPairs=2, roiPath=roi_path
        )

        measurement.assert_called_once()
        np.testing.assert_array_equal(result, [12, 12])
        self.assertEqual(np.loadtxt(os.path.join(self.temp_dir, 'anSizeAll.csv')).shape, (2, 400))


class VideoPropertiesTests(VideoTestCase):
    def test_uncached_and_cached_metadata_match_without_debug_output(self):
        avi_path = os.path.join(self.temp_dir, 'video.avi')
        # Keep the existing ffprobe byte parsing, including its trailing terminator.
        probe_output = (b'[STREAM]\r\nwidth=512\r\nheight=256\r\navg_frame_rate=60/2\r\n'
                        b'[/FORMAT]\r\n')
        process = self.patch('functions.getVideoProperties.subprocess.Popen')
        process.return_value.communicate.return_value = (probe_output, b'')

        uncached = video_properties.getVideoProperties(avi_path)
        cached = video_properties.getVideoProperties(avi_path)

        self.assertEqual(uncached, {'width': '512', 'height': '256',
                                   'avg_frame_rate': '60/2', 'fps': 30})
        self.assertEqual(cached, uncached)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, 'videoProps.pickle')))
        process.assert_called_once_with(
            ['ffprobe', '-show_format', '-show_streams', '-pretty', '-loglevel', 'quiet', avi_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(self.stdout.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
