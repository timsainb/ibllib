import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from oneibl.one import ONE
from ibllib.qc.camera import CameraQC
from ibllib.io.raw_data_loaders import load_camera_ssv_times
from ibllib.tests.fixtures import utils
from brainbox.core import Bunch


class TestCameraQC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(
            base_url="https://test.alyx.internationalbrainlab.org",
            username="test_user",
            password="TapetesBloc18",
        )
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        cls.addClassCleanup(matplotlib.use, backend)

    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.session_path = utils.create_fake_session_folder(self.tempdir.name)
        utils.create_fake_raw_video_data_folder(self.session_path)
        self.eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        self.qc = CameraQC(self.session_path, one=self.one, n_samples=5,
                           side='left', stream=False, download_data=False)
        self.qc._type = 'ephys'
        self.addCleanup(plt.close, 'all')

    def test_check_brightness(self):
        self.qc.data['frame_samples'] = self.qc.load_reference_frames('left')
        n = len(self.qc.data['frame_samples'])
        self.qc.frame_samples_idx = np.linspace(0, 1000, n, dtype=int)
        self.assertEqual('PASS', self.qc.check_brightness(display=True))
        # Check plots
        fig = plt.gcf()
        self.assertEqual(3, len(fig.axes))
        expected = np.array([58.07007217, 56.55802917, 46.09558182])
        np.testing.assert_array_almost_equal(fig.axes[0].lines[0]._y, expected)
        # Make frames a third as bright
        self.qc.data['frame_samples'] = (self.qc.data['frame_samples'] / 3).astype(np.int32)
        self.assertEqual('FAIL', self.qc.check_brightness())
        # Change thresholds
        self.qc.data['frame_samples'] = self.qc.load_reference_frames('left')
        self.assertEqual('FAIL', self.qc.check_brightness(bounds=(10, 20)))
        self.assertEqual('FAIL', self.qc.check_brightness(max_std=1e-6))
        # Check outcome when no frame samples loaded
        self.qc.data['frame_samples'] = None
        self.assertEqual('NOT_SET', self.qc.check_brightness())

    def test_check_file_headers(self):
        self.qc.data['video'] = {'fps': 60.}
        self.assertEqual('PASS', self.qc.check_file_headers())
        self.qc.data['video']['fps'] = 150
        self.assertEqual('FAIL', self.qc.check_file_headers())
        self.qc.data['video'] = None
        self.assertEqual('NOT_SET', self.qc.check_file_headers())

    def test_check_framerate(self):
        FPS = 60.
        self.qc.data['video'] = {'fps': FPS}
        self.qc.data['timestamps'] = np.array([round(1 / FPS, 4)] * 1000).cumsum()
        outcome, frate = self.qc.check_framerate()
        self.assertEqual('PASS', outcome)
        self.assertEqual(59.88, frate)
        self.assertEqual('FAIL', self.qc.check_framerate(threshold=1e-2)[0])
        self.qc.data['timestamps'] = None
        self.assertEqual('NOT_SET', self.qc.check_framerate())

    def test_check_pin_state(self):
        FPS = 60.
        self.assertEqual('NOT_SET', self.qc.check_pin_state())
        # Add some dummy data
        self.qc.data.timestamps = np.array([round(1 / FPS, 4)] * 5).cumsum()
        self.qc.data.pin_state = np.zeros_like(self.qc.data.timestamps, dtype=int)
        self.qc.data.pin_state[1:-1] = 10000
        self.qc.data['video'] = {'fps': FPS, 'length': len(self.qc.data.timestamps)}
        self.qc.data.audio = self.qc.data.timestamps[[0, -1]] - 10e-3

        # Check passes and plots results
        outcome, *_ = self.qc.check_pin_state(display=True)
        self.assertEqual('PASS', outcome)
        a, b = [ln.get_xdata() for ln in plt.gcf().axes[0].lines]
        self.assertEqual(a, self.qc.data.timestamps[1])
        np.testing.assert_array_equal(b, self.qc.data.audio)

        # Fudge some numbers
        self.qc.data.pin_state[2] = 11e3
        self.assertEqual('WARNING', self.qc.check_pin_state()[0])
        self.qc.data['video']['length'] = 10
        outcome, *dTTL = self.qc.check_pin_state()
        self.assertEqual('FAIL', outcome)
        self.assertEqual([0, -5], dTTL)

    def test_check_dropped_frames(self):
        n = 20
        self.qc.data.count = np.arange(n)
        self.qc.data.video = {'length': n}
        self.assertEqual('PASS', self.qc.check_dropped_frames()[0])

        # Drop some frames
        dropped = 6
        self.qc.data.count = np.append(self.qc.data.count, n + dropped)
        outcome, dframe, sz_diff = self.qc.check_dropped_frames()
        self.assertEqual('FAIL', outcome)
        self.assertEqual(dropped, dframe)
        self.assertEqual(1, sz_diff)

        # Verify threshold arg; should be warning due to size diff
        outcome, *_ = self.qc.check_dropped_frames(threshold=70.)
        self.assertEqual('WARNING', outcome)

        # Verify critical outcome
        self.qc.data.count = np.random.permutation(self.qc.data.count)  # Count out of order
        self.assertEqual('CRITICAL', self.qc.check_dropped_frames([0]))

        # Verify not set outcome
        self.qc.data.video = None
        self.assertEqual('NOT_SET', self.qc.check_dropped_frames())

    def test_check_focus(self):
        self.qc.side = 'left'
        self.qc.frame_samples_idx = np.linspace(0, 100, 20, dtype=int)
        outcome = self.qc.check_focus(test=True, display=True)
        self.assertEqual('FAIL', outcome)
        # Verify figures
        figs = plt.get_fignums()
        self.assertEqual(len(plt.figure(figs[0]).axes), 16)
        # Verify Laplacian on blurred images
        expected = np.array([13.19, 14.24, 15.44, 16.64, 18.67, 21.51, 25.99, 31.77,
                             40.75, 52.52, 71.12, 98.26, 149.85, 229.96, 563.53, 563.53])
        actual = [round(x, 2) for x in plt.figure(figs[1]).axes[3].lines[0]._y.tolist()]
        np.testing.assert_array_equal(expected, actual)
        # Verify fft on blurred images
        expected = np.array([6.91, 7.2, 7.61, 8.08, 8.76, 9.47, 10.35, 11.22,
                             11.04, 11.42, 11.35, 11.94, 12.45, 13.22, 13.6, 13.6])
        actual = [round(x, 2) for x in plt.figure(figs[2]).axes[3].lines[0]._y.tolist()]
        np.testing.assert_array_equal(expected, actual)

        # Verify not set outcome
        outcome = self.qc.check_focus()
        self.assertEqual('NOT_SET', outcome)

        # Verify ROI
        self.qc.data.frame_samples = self.qc.load_reference_frames('left')
        outcome = self.qc.check_focus(roi=None)
        self.assertEqual('PASS', outcome)

    def test_check_position(self):
        # Verify test mode
        outcome = self.qc.check_position(test=True, display=True)
        self.assertEqual('PASS', outcome)

        # Verify plots
        axes = plt.gcf().axes
        self.assertEqual(3, len(axes))
        expected = np.array([100., 93.74829841, 93.2494463])
        np.testing.assert_almost_equal(axes[2].lines[0]._y, expected)

        # Verify not set (no frame samples and not in test mode)
        outcome = self.qc.check_position()
        self.assertEqual('NOT_SET', outcome)

        # Verify percent threshold as False
        thresh = (75, 80)
        outcome = self.qc.check_position(test=True, pct_thresh=False,
                                         hist_thresh=thresh, display=True)
        self.assertEqual('FAIL', outcome)
        fig = plt.get_fignums()[-1]
        thr = [ln._y[0] for ln in plt.figure(fig).axes[2].lines[1:]]
        self.assertCountEqual(thr, thresh, 'unexpected thresholds in figure')

    def test_check_resolution(self):
        self.qc.data['video'] = {'width': 1280, 'height': 1024}
        self.assertEqual('PASS', self.qc.check_resolution())
        self.qc.data['video']['width'] = 150
        self.assertEqual('FAIL', self.qc.check_resolution())
        self.qc.data['video'] = None
        self.assertEqual('NOT_SET', self.qc.check_resolution())

    def test_check_timestamps(self):
        FPS = 60.
        n = 1000
        self.qc.data['video'] = Bunch({'fps': FPS, 'length': n})
        self.qc.data['timestamps'] = np.array([round(1 / FPS, 4)] * n).cumsum()
        # Verify passes
        self.assertEqual('PASS', self.qc.check_timestamps())
        # Verify fails
        self.qc.data['timestamps'] = np.array([round(1 / 30, 4)] * 100).cumsum()
        self.assertEqual('FAIL', self.qc.check_timestamps())
        # Verify not set
        self.qc.data['video'] = None
        self.assertEqual('NOT_SET', self.qc.check_timestamps())

    def test_check_camera_times(self):
        outcome = self.qc.check_camera_times()
        self.assertEqual('NOT_SET', outcome)

        # Verify passes
        self.qc.side = 'body'
        ts_path = Path(__file__).parents[1].joinpath('extractors', 'data', 'session_ephys')
        ssv_times = load_camera_ssv_times(ts_path, self.qc.side)
        self.qc.data.bonsai_times, self.qc.data.camera_times = ssv_times
        self.qc.data.video = Bunch({'length': self.qc.data.bonsai_times.size})

        outcome, _ = self.qc.check_camera_times()
        self.assertEqual('PASS', outcome)

        # Verify warning
        n_over = 14
        self.qc.data.video['length'] -= n_over
        outcome, actual = self.qc.check_camera_times()

        self.assertEqual('WARNING', outcome)
        self.assertEqual(n_over, actual)

    def test_ensure_data(self):
        self.qc.eid = self.eid
        self.qc.download_data = False
        # If data for this session exists locally, overwrite the methods so it is not found
        if self.one.path_from_eid(self.eid).exists():
            self.qc.one.to_eid = lambda _: self.eid
            self.qc.one.download_datasets = lambda _: None
        with self.assertRaises(AssertionError):
            self.qc.run(update=False)

    def tearDown(self) -> None:
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
