"""The widefield data extraction pipeline.

The widefield pipeline requires task data extraction using the FPGA (ephys_preprocessing),
optogenetics, camera extraction and widefield image data compression, SVD and correction.
"""
import logging
from collections import OrderedDict
import traceback
from pathlib import Path
import packaging.version

import numpy as np
import pandas as pd

import one.alf.io as alfio

from ibllib.misc import check_nvidia_driver
from ibllib.ephys import ephysqc, spikes, sync_probes
from ibllib.io import ffmpeg, spikeglx
from ibllib.io.video import label_from_path
from ibllib.io.extractors.widefield import Widefield as WidefieldExtractor
from ibllib.pipes import tasks
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.pipes.ephys_preprocessing import (
    EphysPulses, EphysMtscomp, EphysAudio, EphysVideoCompress, EphysVideoSyncQc, EphysTrials, EphysPassive, EphysDLC, EphysPostDLC
)
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC

_logger = logging.getLogger('ibllib')


#  level 1
class WidefieldPreprocess(tasks.Task):
    priority = 60
    level = 1
    force = False
    signature = {
        'input_files': [('widefield.raw.*', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.*', 'raw_widefield_data', True),
                        ('widefieldLandmarks.dorsalCortex.*', 'raw_widefield_data*', True)],
        'output_files': [('*trials.choice.npy', 'alf', True), ]
    }

    def _run(self):
        dsets, out_files = WidefieldExtractor(self.session_path).extract(save=True)

        if not self.one or self.one.offline:
            return out_files

        # Run the task QC
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one)
        # Extract extra datasets required for QC
        qc.extractor.data = dsets
        qc.extractor.extract_data()
        # Aggregate and update Alyx QC fields
        qc.run(update=True)
        return out_files


def _extract_haemo_corrected():
    U = np.load('U.npy')
    SVT = np.load('SVT.npy')

    frame_rate = 30.  # acquisition rate (2 channels)
    output_folder = None  # write to current directory or path

    from wfield.ncaas import dual_color_hemodymamic_correction

    SVTcorr = dual_color_hemodymamic_correction(U, SVTa, SVTb, frame_rate=frame_rate, output_folder=output_folder);

# pipeline
class WidefieldExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(WidefieldExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = EphysPulses(self.session_path)
        # tasks["EphysRawQC"] = RawEphysQC(self.session_path)
        tasks["EphysAudio"] = EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        # level 1
        # tasks["SpikeSorting"] = SpikeSorting(
        #     self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysTrials"] = EphysTrials(self.session_path, parents=[tasks["EphysPulses"]])

        tasks["EphysPassive"] = EphysPassive(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysTrials"]])
        # tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        # level 3
        tasks["EphysPostDLC"] = EphysPostDLC(self.session_path, parents=[tasks["EphysDLC"]])
        self.tasks = tasks
