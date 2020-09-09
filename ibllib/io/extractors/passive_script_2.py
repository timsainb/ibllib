#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, September 7th 2020, 11:51:17 am
import alf.io
from oneibl.one import ONE
from pathlib import Path
import numpy as np
import json
import ibllib.io.extractors.passive as passive
from ibllib.io.extractors import ephys_fpga
import ibllib.io.raw_data_loaders as rawio
from ibllib.qc.oneutils import random_ephys_session

# hardcoded var
FRAME_FS = 60  # Sampling freq of the ipad screen, in Hertz
FS_FPGA = 30000  # Sampling freq of the neural recording system screen, in Hertz
NVALVE = 40  # number of expected valve clicks
NGABOR = 20 + 20 * 4 * 2  # number of expected Gabor patches
NTONES = 40
NNOISES = 40
DEBUG_PLOTS = False

# load data
one = ONE()
dataset_types = [
    "_spikeglx_sync.times",
    "_spikeglx_sync.channels",
    "_spikeglx_sync.polarities",
    "_iblrig_RFMapStim.raw",
    "_iblrig_stimPositionScreen.raw",
    "_iblrig_syncSquareUpdate.raw",
    "ephysData.raw.meta",
    "_iblrig_taskSettings.raw",
    "_iblrig_taskData.raw",
]

eid = "01864d6f-31e8-49c9-aadd-2e5021ea0ee7"  # not working
# number of expected spacers wrong
eid = "fff7d745-bbce-4756-a690-3431e2f3d108"
eid = "849c9acb-8223-4e09-8cb1-95004b452baf"
# AssertionError: multiple object sync with the same attribute in probe01, restrict parts/namespace
eid = "c7a9f89c-2c1d-4302-94b8-effcbe4a85b3"
# OK
eid = "193fe7a8-4eb5-4f3e-815a-0c45864ddd77"  # HIGH
# eid = one.search(subject="CSH_ZAD_022", date_range="2020-05-24", number=1)[0]
eid = "a82800ce-f4e3-4464-9b80-4c3d6fade333"  # LOW

# eid, det = random_ephys_session()

local_paths = one.load(eid, dataset_types=dataset_types, download_only=True)

session_path = alf.io.get_session_path(local_paths[0])


# load session fixtures
def load_passive_session_fixtures(session_path, meta=None):
    settings = rawio.load_settings(session_path)
    ses_nb = settings["SESSION_ORDER"][settings["SESSION_IDX"]]
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")

    fixture = {
        "pcs": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_pcs.npy")),
        "delays": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimDelays.npy")),
        "ids": np.load(path_fixtures.joinpath(f"session_{ses_nb}_passive_stimIDs.npy")),
    }

    return fixture


def load_passive_stim_meta():
    path_fixtures = Path(ephys_fpga.__file__).parent.joinpath("ephys_sessions")
    with open(path_fixtures.joinpath("passive_stim_meta.json"), "r") as f:
        meta = json.load(f)

    return meta
# load general metadata

# fpga_sync = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"])


def get_passive_spacers(session_path, sync=None, sync_map=None):
    """
    load and get spacer information, do corr to find spacer timestamps
    returns t_passive_starts, t_starts, t_ends
    """
    if sync is None or sync_map is None:
        sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)

    meta = load_passive_stim_meta()
    t_end_ephys = passive.ephysCW_end(session_path=session_path)
    fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=t_end_ephys)
    spacer_template = (
        np.array(meta["VISUAL_STIM_0"]["ttl_frame_nums"], dtype=np.float32) / FRAME_FS
    )
    jitter = 3 / FRAME_FS  # allow for 3 screen refresh as jitter
    t_quiet = meta["VISUAL_STIM_0"]["delay_around"]
    spacer_times, _ = passive.get_spacer_times(
        spacer_template=spacer_template, jitter=jitter, ttl_signal=fttl["times"], t_quiet=t_quiet
    )

    # Check correct number of spacers found
    n_exp_spacer = np.sum(np.array(meta["STIM_ORDER"]) == 0)  # Hardcoded 0 for spacer
    if n_exp_spacer != np.size(spacer_times) / 2:
        raise ValueError(
            f"The number of expected spacer ({n_exp_spacer}) "
            f"is different than the one found on the raw "
            f"trace ({np.size(spacer_times)/2})"
        )

    spacer_times = np.r_[spacer_times.flatten(), sync["times"][-1]]
    return spacer_times[0], spacer_times[1::2], spacer_times[2::2]


# Load sessions sync channels, map adnd fixtures
sync, sync_map = ephys_fpga._get_main_probe_sync(session_path, bin_exists=False)
fixture = load_passive_session_fixtures(session_path)

# Define start and end times of the 3 passive periodes
t_start_passive, t_starts, t_ends = get_passive_spacers(session_path, sync=sync, sync_map=sync_map)
tspontaneous = [t_starts[0], t_ends[0]]
trfm = [t_starts[1], t_ends[1]]
treplay = [t_starts[2], t_ends[2]]
# TODO export this to a dstype




# 3/3 Replay of task stimuli
fttl = ephys_fpga._get_sync_fronts(sync, sync_map["frame2ttl"], tmin=treplay[0])
audio = ephys_fpga._get_sync_fronts(sync, sync_map["audio"], tmin=treplay[0])

# get idxs of where the diff is of a gabor presentation.
# This will get the start of a gabor patch presentation
# and ignore the first pulse where the onset is hidden
# 0.3 is the expected gabor length and 0.5 isthe expected delay length.
# We use 0.4 to split the difference and to allow for maximum drift
# At this stage we want to define what pulses are and not quality control them.
diff_idxs = np.where(np.diff(fttl['times']) < 0.4)[0]
# move one change back, i.e. get the OFFset of the previous stimulus
# get the previous polarity change (which should be the end of previous stim presentation)
idx_end_stim = diff_idxs - 1
# We've now lost the last stim presentation so get the last onset and move to it's offset
# append it to the end indexes diff_idx[-1] + 1
idx_end_stim = np.append(idx_end_stim, diff_idxs[-1] + 1)
assert len(idx_end_stim) == sum(fixture['ids'] == 'G'), "wrong number of GaborEnd times"
# np.median(np.diff(fttl['times'])[diff_idxs])

# Get the start times from the end times
start_times = fttl['times'][idx_end_stim - 1]
# patch the first stim onset time that is wrong
if fttl['times'][idx_end_stim[0]] - fttl['times'][idx_end_stim[0]-1]  > 0.3:
    start_times[0] = fttl['times'][idx_end_stim[0]] - 0.3
# Move the end times to a var
end_times = fttl['times'][idx_end_stim]

passiveGabor_properties = fixture['pcs']
passiveGabor_properties_metadata = ['position, contrast, phase']
# intervals dstype requires reshaping of start and end times
passiveGabor_intervals = np.array([(x,y) for x, y in zip(start_times, end_times)])

# Check length of presentation of stim is  within 100msof expected
np.allclose(np.array([y-x for x, y in passiveGabor_intervals]), 0.3, 0.1)
# passiveValve.intervals

# Get valve intervals from bpod channel
bpod = ephys_fpga._get_sync_fronts(sync, sync_map["bpod"], tmin=treplay[0])
len(bpod['times']) == NVALVE * 2  # (40 * 2)
# Get Tone and Noise cue instervals

# Get Gabor patches intervals

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ibllib.plots import squares, vertical_lines, color_cycle


plt.plot(np.diff(fttl['times']), '.')

plt.axhline(0.4)
# plt.axvline(fttl['times'][idx_end_stim])


pl, ax = plt.subplots(1, 1)
for i, lab in enumerate(["frame2ttl", "audio", "bpod"]):
    sy = ephys_fpga._get_sync_fronts(sync, sync_map[lab], tmin=t_start_passive)
    squares(sy["times"], sy["polarities"], yrange=[0.1 + i, 0.9 + i], color="k", ax=ax)

vertical_lines(
    np.r_[t_start_passive, t_starts, t_ends],
    ymin=-1,
    ymax=4,
    color=color_cycle(0),
    ax=ax,
    label="spacers",
)
vertical_lines(
    start_times,
    ymin=0,
    ymax=1,
    color=color_cycle(1),
    ax=ax,
    label="Gabor start times",
)
vertical_lines(
    end_times,
    ymin=0,
    ymax=1,
    color=color_cycle(2),
    ax=ax,
    label="Gabor end times",
)

ax.legend()
# plt.show()
# %gui qt
# print(det)
print(eid)
