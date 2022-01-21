"""Data extraction from widefield binary file"""
from collections import OrderedDict
import logging
from pathlib import Path, PureWindowsPath
import uuid

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import parse_version
# from wfield.decomposition import approximate_svd
# from wfield.plots import plot_summary_motion_correction
# from wfield.registration import motion_correct
from wfield import decomposition, plots, registration, utils, io as wfield_io

import one.alf.io as alfio
from iblutil.util import Bunch
import ibllib.dsp as dsp
import ibllib.exceptions as err
from ibllib.io.raw_data_loaders import load_widefield_mmap
from ibllib.io.extractors import biased_trials, training_trials
from ibllib.io.extractors.base import BaseExtractor
from ibllib.io.extractors.training_wheel import extract_wheel_moves

_logger = logging.getLogger('ibllib')


class Widefield(BaseExtractor):
    save_names = ('_ibl_trials.feedbackType.npy',)
    var_names = ('feedbackType',)

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)

    def _extract(self, **kwargs):
        """
        NB: kwargs should be loaded from meta file
        Parameters
        ----------
        n_channels
        dtype
        shape
        kwargs

        Returns
        -------

        """
        self.preprocess(**kwargs)
        ##########################################################
        # dat = load_widefield_mmap(self.session_path, dtype=dtype, shape=shape, mode='r+')

        return [out[k] for k in out] + [wheel['timestamps'], wheel['position'],
                                        moves['intervals'], moves['peakAmplitude']]

    def preprocess(self, fs=30, functional_channel=0, nbaseline_frames=30, k=200):
        from wfield.cli import _motion, _baseline, _decompose, _hemocorrect, load_stack
        data_path = self.session_path.joinpath('raw_widefield_data')

        # MOTION CORRECTION
        _motion(data_path)
        # COMPUTE AVERAGE FOR BASELINE
        _baseline(data_path, nbaseline_frames)
        # DATA REDUCTION
        _decompose(data_path, k=k)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = load_stack(data_path)
        if dat.shape[1] == 2:
            del dat
            _hemocorrect(data_path, fs=fs, functional_channel=functional_channel)

    def rename_files(session_folder) -> bool:
        """
        Rename the raw widefield data for a given session.

        Parameters
        ----------
        session_folder : str, pathlib.Path
            A session path containing widefield data.

        Returns
        -------
        success : bool
            True if all files were successfully renamed.
        TODO Double-check filenames and call this function
        """
        session_path = Path(session_folder).joinpath('raw_widefield_data')
        if not session_path.exists():
            _logger.warning(f'Path does not exist: {session_path}')
            return False
        renames = (
            ('dorsal_cortex_landmarks.json', 'widefieldLandmarks.dorsalCortex.json'),
            ('*.dat', 'widefield.raw.dat'),
            ('*.camlog', 'widefieldEvents.raw.camlog')
        )
        success = True
        for before, after in renames:
            try:
                filename = next(session_path.glob(before))
                filename.rename(after)
                # TODO Save nchannels and frame size from filename?
            except StopIteration:
                _logger.warning(f'File not found: {before}')
                success = False
        return success
