from pathlib import Path
import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np

from ibllib.ephys.neuropixel import SITES_COORDINATES
from oneibl.one import ONE
import alf.io
import ibllib.atlas as atlas
from ibllib.ephys.spikes import probes_description as extract_probes

_logger = logging.getLogger('ibllib')
atlas_params = {
    'PATH_ATLAS': str('/datadisk/BrainAtlas/ATLASES/Allen'),
    'FILE_REGIONS':
        str(Path(inspect.getfile(atlas.AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
    'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
}
# origin Allen left, front, up
brain_atlas = atlas.AllenAtlas(res_um=25, par=atlas_params)


def load_track_csv(file_track):
    """
    Loads a lasagna track and convert to IBL-ALlen coordinate framework
    :param file_track:
    :return: xyz
    """
    # apmldv in the histology file is flipped along x and y directions
    ixiyiz = np.loadtxt(file_track, delimiter=',')[:, [1, 0, 2]]
    ixiyiz[:, 1] = 527 - ixiyiz[:, 1]
    ixiyiz = ixiyiz[np.argsort(ixiyiz[:, 2]), :]
    xyz = brain_atlas.bc.i2xyz(ixiyiz)
    xyz[:, 0] = - xyz[:, 0]
    return xyz


def get_picked_tracks(histology_path, glob_pattern="*_pts_transformed.csv"):
    """
    This outputs reads in the Lasagna output and converts the picked tracks in the IBL coordinates
    :param histology_path: Path object: folder path containing tracks
    :return: xyz coordinates in
    """
    xyzs = []
    histology_path = Path(histology_path)
    if histology_path.is_file():
        files_track = [histology_path]
    else:
        files_track = list(histology_path.rglob(glob_pattern))
    for file_track in files_track:
        xyzs.append(load_track_csv(file_track))
    return {'files': files_track, 'xyz': xyzs}


def get_micro_manipulator_data(subject, one=None, force_extract=False):
    """
    Looks for all ephys sessions for a given subject and get the probe micro-manipulator
    trajectories.
    If probes ALF object not on flat-iron, attempts to perform the extraction from meta-data
    and task settings file.
    """
    if not one:
        one = ONE()

    eids, sessions = one.search(subject=subject, task_protocol='ephys', details=True)
    probes = alf.io.AlfBunch({})
    for ses in sessions:
        sess_path = Path(ses['local_path'])
        probe = None
        if not force_extract:
            probe = one.load_object(ses['url'], 'probes')
        if not probe:
            _logger.warning(f"Re-extraction probe info for {sess_path}")
            dtypes = ['_iblrig_taskSettings.raw', 'ephysData.raw.meta']
            raw_files = one.load(ses['url'], dataset_types=dtypes, download_only=True)
            if all([rf is None for rf in raw_files]):
                _logger.warning(f"no raw settings files nor ephys data found for"
                                f" {ses['local_path']}. Skip this session.")
                continue
            extract_probes(sess_path, bin_exists=False)
            probe = alf.io.load_object(sess_path.joinpath('alf'), 'probes')
        one.load(ses['url'], dataset_types='channels.localCoordinates', download_only=True)
        # get for each insertion the sites local mapping: if not found assumes checkerboard pattern
        probe['sites_coordinates'] = []
        for prb in probe.description:
            chfile = Path(ses['local_path']).joinpath('alf', prb['label'],
                                                      'channels.localCoordinates.npy')
            if chfile.exists():
                probe['sites_coordinates'].append(np.load(chfile))
            else:
                _logger.warning(f"no channel.localCoordinates found for {ses['local_path']}."
                                f"Assumes checkerboard pattern")
                probe['sites_coordinates'].append(SITES_COORDINATES)
        # put the session information in there
        probe['session'] = [ses] * len(probe.description)
        probes = probes.append(probe)
    return probes


def plot_merged_result(probes, tracks, index=None):
    pass
    # fig, ax = plt.subplots(1, 2)
    # # plot the atlas image
    # brain_atlas.plot_cslice(np.mean(xyz[:, 1]) * 1e3, ax=ax[0])
    # brain_atlas.plot_sslice(np.mean(xyz[:, 0]) * 1e3, ax=ax[1])
    # # plot the full tracks
    # ax[0].plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3)
    # ax[1].plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3)
    # # plot the sites
    # ax[0].plot(xyz_channels[:, 0] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')
    # ax[1].plot(xyz_channels[:, 1] * 1e3, xyz_channels[:, 2] * 1e3, 'y*')


def plot2d_all(trajectories, tracks):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    plt.figure()
    axs = brain_atlas.plot_sslice(brain_atlas.bc.i2x(190) * 1e3, cmap=plt.get_cmap('bone'))
    plt.figure()
    axc = brain_atlas.plot_cslice(brain_atlas.bc.i2y(350) * 1e3)
    for xyz in tracks['xyz']:
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3, 'b')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3, 'b')
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        xyz = ins.xyz
        axc.plot(xyz[:, 0] * 1e3, xyz[:, 2] * 1e3, 'r')
        axs.plot(xyz[:, 1] * 1e3, xyz[:, 2] * 1e3, 'r')


def plot3d_all(trajectories, tracks):
    """
    Plot all tracks on a single 2d slice
    :param trajectories: dictionary output of the Alyx REST query on trajectories
    :param tracks:
    :return:
    """
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(brain_atlas.label)
    mlab.pipeline.iso_surface(src, contours=[0.5, ], opacity=0.3)

    pts = []
    for xyz in tracks['xyz']:
        mlapdv = brain_atlas.bc.xyz2i(xyz)
        pts.append(mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2], line_width=3))

    plt_trj = []
    for trj in trajectories:
        ins = atlas.Insertion.from_dict(trj, brain_atlas=brain_atlas)
        mlapdv = brain_atlas.bc.xyz2i(ins.xyz)
        plt = mlab.plot3d(mlapdv[:, 1], mlapdv[:, 0], mlapdv[:, 2],
                          line_width=3, color=(1., 0., 1.))
        plt_trj.append(plt)


def get_brain_regions(xyz, channels_positions=SITES_COORDINATES, brain_atlas=brain_atlas):
    """
    :param xyz: numpy array of 3D coordinates corresponding to a picked track or a trajectory
    the deepest point is assumed to be the tip.
    :param channels_positions:
    :param brain_atlas:
    :return:
    """

    """
    this is the depth along the probe (from the first point which is the deepest labeled point)
    Due to the blockiness, depths may not be unique along the track so it has to be prepared
    """
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    ind_depths = np.argsort(d)
    d = np.sort(d)
    iid = np.where(np.diff(d) >= 0)[0]
    ind_depths = ind_depths[iid]
    d = d[iid]

    """
    Interpolate channel positions along the probe depth and get brain locations
    """
    xyz_channels = np.zeros((channels_positions.shape[0], 3))
    for m in np.arange(3):
        xyz_channels[:, m] = np.interp(channels_positions[:, 1] / 1e6,
                                       d[ind_depths], xyz[ind_depths, m])
    brain_regions = brain_atlas.regions.get(brain_atlas.get_labels(xyz_channels))
    brain_regions['xyz'] = xyz_channels
    brain_regions['lateral'] = channels_positions[:, 0]
    brain_regions['axial'] = channels_positions[:, 1]
    assert np.unique([len(brain_regions[k]) for k in brain_regions]).size == 1
    """
    Get the probe insertion from the coordinates
    """
    insertion = atlas.Insertion.from_track(xyz, brain_atlas)
    return brain_regions, insertion


def register_track(probe_id, picks=None, one=None):
    """
    Register the user picks to a probe in Alyx
    Here we update Alyx models on the database in 3 steps
    1) The user picks converted to IBL coordinates will be stored in the json field of the
    corresponding probe insertion models
    2) The trajectory computed from the histology track is created or patched
    3) Channel locations are set in the table
    """
    assert one
    brain_locations, insertion_histology = get_brain_regions(picks)
    # 1) update the alyx models, first put the picked points in the insertion json
    one.alyx.rest('insertions', 'partial_update',
                  id=probe_id,
                  data={'json': {'xyz_picks': np.int32(picks * 1e6).tolist()}})
    # TODO: add the affine transform parameters

    # 2) patch or create the trajectory coming from histology track
    tdict = {'probe_insertion': probe_id,
             'x': insertion_histology.x * 1e6,
             'y': insertion_histology.y * 1e6,
             'z': insertion_histology.z * 1e6,
             'phi': insertion_histology.phi,
             'theta': insertion_histology.theta,
             'depth': insertion_histology.depth * 1e6,
             'roll': insertion_histology.beta,
             'provenance': 'Histology track',
             'coordinate_system': 'IBL-Allen',
             }
    hist_traj = one.alyx.rest('trajectories', 'list',
                              probe_insertion=probe_id,
                              provenance='Histology track')
    # if the trajectory exists, remove it, this will cascade delete existing channel locations
    if len(hist_traj):
        one.alyx.rest('trajectories', 'delete', id=hist_traj[0]['id'])
    hist_traj = one.alyx.rest('trajectories', 'create', data=tdict)

    # 3) create channel locations
    channel_dict = []
    for i in np.arange(brain_locations.id.size):
        channel_dict.append({
            'x': brain_locations.xyz[i, 0] * 1e6,
            'y': brain_locations.xyz[i, 1] * 1e6,
            'z': brain_locations.xyz[i, 2] * 1e6,
            'axial': brain_locations.axial[i],
            'lateral': brain_locations.lateral[i],
            'brain_region': int(brain_locations.id[i]),
            'trajectory_estimate': hist_traj['id']
        })
    one.alyx.rest('channels', 'create', data=channel_dict)
