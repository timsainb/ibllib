from pathlib import Path
from oneibl.one import ONE
import alf.io
import numpy as np
one = ONE()
import json
import re
import os
import matplotlib.pyplot as plt
import cv2
from os.path import join
import pywt

def load_animals_of_interest(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container][0]
    print(data.shape)
    return data

def load_dlc(folder_path):
    """
    Load DLC features
    :param folder_path: path to DLC features
    :return: DLC features as an array, list of DLC features, dictionary of DLC features and times for these
    """

    # Load in DLC data
    dlc_features = np.load(join(folder_path, 'alf', '_ibl_leftCamera.dlc.npy'))
    camera_times = np.load(join(folder_path, 'alf', '_ibl_leftCamera.times.npy'))
    json1_file = open(join(folder_path, 'alf', '_ibl_leftCamera.dlc.metadata.json'))
    json1_str = json1_file.read()
    dlc_meta = json.loads(json1_str)['columns']

    # Create DLC dictionary:
    dlc_dict ={}
    for j, feat in enumerate(dlc_meta):
        dlc_dict[feat] = dlc_features[:, j]

    # check order
    assert dlc_meta[0] == 'pupil_top_r_x', 'DLC feature order is off!'
    assert dlc_meta[11] == 'pupil_left_r_likelihood', 'DLC feature order is off!'
    return dlc_features, dlc_meta, dlc_dict, camera_times


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_mean_std_around_event(event, diameter, times, eid, trials):
    '''
    event in {'stimOn_times', 'feedback_times', 'stimOff_times'}
    '''
    event_times = trials[event]

    window_size = 70

    # segments = []
    fig, ax = plt.subplots()
    # skip first and last trials to get same window length
    for t in event_times[5:-5]:
        idx = find_nearest(times, t)
        segments = diameter[idx - window_size: idx + window_size]
        plt.plot(range(len(segments)), segments, linewidth=3, alpha=0.1)
    plt.ylabel('pupil diameter [px]')
    plt.xlabel('frames')
    plt.title(eid)
    plt.tight_layout()

    M = np.nanmean(np.array(segments), axis=0)
    E = np.nanstd(np.array(segments), axis=0)

def get_pupil_diameter(alf_path):

    json1_file = open(alf_path / '_ibl_leftCamera.dlc.metadata.json')
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)['columns']

    # check order
    assert json1_data[0] == 'pupil_top_r_x', 'Order is off!'
    assert json1_data[11] == 'pupil_left_r_likelihood', 'Order is off!'

    dlc = np.load(alf_path / '_ibl_leftCamera.dlc.npy')

    K = {}
    K['pupil_top_r'] = dlc[:, :3]
    K['pupil_right_r'] = dlc[:, 3:6]
    K['pupil_bottom_r'] = dlc[:, 6:9]
    K['pupil_left_r'] = dlc[:, 9:12]

    # Set values to nan if likelyhood is too low
    XYs = {}
    for part in K:
        x = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 0])
        x = x.filled(np.nan)
        y = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 1])
        y = y.filled(np.nan)
        XYs[part] = [x, y]

    # get both diameters (d1 = top - bottom, d2 = left - right)
    d1 = ((XYs['pupil_top_r'][0] - XYs['pupil_bottom_r'][0])**2 +
          (XYs['pupil_top_r'][1] - XYs['pupil_bottom_r'][1])**2)**0.5
    d2 = ((XYs['pupil_left_r'][0] - XYs['pupil_right_r'][0])**2 +
          (XYs['pupil_left_r'][1] - XYs['pupil_right_r'][1])**2)**0.5
    d = np.mean([d1, d2], axis=0)

    return d

def add_stim_off_times(trials):
    """
    add stim_off times to trials dictionary. Calculate this by adding 1 to feedback_times for correct trials and by adding 2 to feedback_times for inccorect trials
    :param trials: dictionary of trial info from the rig
    :return: trials dictionary with an entry for stim_off.  stimOff time
    """

    on = 'stimOn_times'
    off = 'stimOff_times'
    trials[off] = np.zeros(shape=trials[on].shape)
    correct_trials = trials['feedbackType'] == 1
    u = trials['feedback_times'][correct_trials] + 1.0
    trials[off][correct_trials] = u
    error_trials = trials['feedbackType'] == -1
    v = trials['feedback_times'][error_trials] + 2.0
    trials[off][error_trials] = v

def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
   # print("Frame rate = " + str(fps))
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cap.release()
    return frame_image

def create_frame_array(dlc_data, frame_number):
    """
    Create x_vec, y_vec with paw coordinates from dlc dictionary.  x_vec and y_vec will each have
    four components and will be ordered pinky, ring, middle, pointer
    :param dlc_data: dictionary of DLC features
    :param frame_number: frame number
    :return: vec_x and vec_y
    """
    # Filter out non-position keys
    filtered_dict = {k: dlc_data[k] for k in dlc_data.keys() if k.endswith(('x', 'y'))}
    keys = set(map(lambda k: k[0:-2], filtered_dict.keys()))
    xy = [(filtered_dict[k + '_x'][frame_number], filtered_dict[k + '_y'][frame_number])
          for k in keys]
    return xy, filtered_dict.keys()

def calculate_diameter(dlc):
    K = {}
    K['pupil_top_r'] = dlc[:, :3]
    K['pupil_right_r'] = dlc[:, 3:6]
    K['pupil_bottom_r'] = dlc[:, 6:9]
    K['pupil_left_r'] = dlc[:, 9:12]

    # Set values to nan if likelyhood is too low
    XYs = {}
    for part in K:
        x = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 0])
        x = x.filled(np.nan)
        y = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 1])
        y = y.filled(np.nan)
        XYs[part] = [x, y]

    # get both diameters (d1 = top - bottom, d2 = left - right)
    d1 = ((XYs['pupil_top_r'][0] - XYs['pupil_bottom_r'][0]) ** 2 +
          (XYs['pupil_top_r'][1] - XYs['pupil_bottom_r'][1]) ** 2) ** 0.5
    d2 = ((XYs['pupil_left_r'][0] - XYs['pupil_right_r'][0]) ** 2 +
          (XYs['pupil_left_r'][1] - XYs['pupil_right_r'][1]) ** 2) ** 0.5
    d = np.mean([d1, d2], axis=0)
   # print(np.array([XYs['pupil_top_r'][0], XYs['pupil_bottom_r'][0], XYs['pupil_left_r'][0], XYs['pupil_right_r'][0]]).shape)
    x_circ = np.mean([XYs['pupil_top_r'][0], XYs['pupil_bottom_r'][0], XYs['pupil_left_r'][0], XYs['pupil_right_r'][0]], axis = 0)
    y_circ = np.mean([XYs['pupil_top_r'][1], XYs['pupil_bottom_r'][1],XYs['pupil_left_r'][1], XYs['pupil_right_r'][1]], axis = 0)
    return x_circ, y_circ, d

def denoise_signal(z):
    coeffs = pywt.wavedec(z, 'db4', level=3)
    # remove highest frequency signals
    coeffs[3] = coeffs[3] * 0
    coeffs[2] = coeffs[2] * 0
    reconstructed_signal = pywt.waverec(coeffs, 'db4')
    return reconstructed_signal

def denoise_pupil_calculate_diameter(dlc):
    K = {}
    K['pupil_top_r'] = dlc[:, :3]
    K['pupil_right_r'] = dlc[:, 3:6]
    K['pupil_bottom_r'] = dlc[:, 6:9]
    K['pupil_left_r'] = dlc[:, 9:12]

    # Set values to nan if likelyhood is too low
    XYs = {}
    for part in K:
        x = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 0])
        x = x.filled(np.nan)
        # Denoise by applying wavelet transform:
        y = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 1])
        y = y.filled(np.nan)
        # Denoise by applying wavelet transform:
        y = denoise_signal(y)[:len(y)]
        XYs[part] = [x, y]

    # get both diameters (d1 = top - bottom, d2 = left - right)
    d1 = ((XYs['pupil_top_r'][0] - XYs['pupil_bottom_r'][0]) ** 2 +
          (XYs['pupil_top_r'][1] - XYs['pupil_bottom_r'][1]) ** 2) ** 0.5
    d2 = ((XYs['pupil_left_r'][0] - XYs['pupil_right_r'][0]) ** 2 +
          (XYs['pupil_left_r'][1] - XYs['pupil_right_r'][1]) ** 2) ** 0.5
    d = np.mean([d1, d2], axis=0)
    # print(np.array([XYs['pupil_top_r'][0], XYs['pupil_bottom_r'][0], XYs['pupil_left_r'][0], XYs['pupil_right_r'][0]]).shape)
    x_circ = np.mean([XYs['pupil_top_r'][0], XYs['pupil_bottom_r'][0], XYs['pupil_left_r'][0], XYs['pupil_right_r'][0]],
                     axis=0)
    y_circ = np.mean([XYs['pupil_top_r'][1], XYs['pupil_bottom_r'][1], XYs['pupil_left_r'][1], XYs['pupil_right_r'][1]],
                     axis=0)
    return x_circ, y_circ, d



def dowload_dlc_data():
    dtypes = ['camera.dlc', 'camera.times',
              '_iblrig_Camera.raw', 'trials.choice',
              'trials.contrastLeft',
              'trials.contrastRight',
              'trials.feedback_times',
              'trials.feedbackType',
              'trials.goCue_times',
              'trials.goCueTrigger_times',
              'trials.probabilityLeft',
              'trials.response_times',
              'trials.rewardVolume',
              'trials.stimOn_times']
    # get eids for sessions with DLC data, raw video data and behavior
    eids = one.search(dataset_types=dtypes)
    print("number of sessions with DLC data = " + str(len(eids)))
    # download data for first eid:
    eid = eids[0]
    D = one.load(eid, dtypes, dclass_output=True)
    saved_path = Path(D.local_path[0]).parent.parent
    return saved_path


if __name__ == "__main__":
    # Set seed
    np.random.seed(0)
    # Download DLC data locally and obtain path to data:
    save_path = dowload_dlc_data()

    # read in DLC features
    dlc_features, dlc_meta, dlc_dict, camera_times = load_dlc(save_path)
    print("DLC features arr shape: " + str(dlc_features.shape)) #(153299, 54) -- there is 153299 time points and 54 features for each time point (27 features and a likelihood value for each)

    #  Path to raw video data
    raw_vid_meta = save_path / 'raw_video_data/_iblrig_leftCamera.raw.mp4'

    # Fit pupil diameter to DLC features
    x_circ, y_circ, michael_d = calculate_diameter(dlc_features)
    x_circ_denoised, y_circ_denoised, michael_d_denoised = denoise_pupil_calculate_diameter(dlc_features)
    assert len(michael_d) == len(michael_d_denoised), "len michael_d = " + str(len(michael_d)) + "; len michael_d_denoised = " + str(len(michael_d_denoised))

    # Read in trials data so as to align DLC features with behavioral events
    trials = alf.io.load_object(save_path / 'alf/', '_ibl_trials')
    add_stim_off_times(trials)

    # get trial number for each time bin
    trial_numbers = np.digitize(camera_times, trials['stimOn_times'])
    print('Range of trials: ', [trial_numbers[0], trial_numbers[-1]])

    # select trial to plot at random:
    n_trials_to_plot = 1
    trials_to_plot = np.random.choice(np.unique(trial_numbers), n_trials_to_plot, replace=False)

    for trial in trials_to_plot:
        dir_for_plotting = str(save_path) + '/explore_raw_data/trial_' + str(trial) + '/'
        if not os.path.exists(dir_for_plotting):
            os.makedirs(dir_for_plotting)
        print("Created directory " + dir_for_plotting + " for saving frames")
        # get frames in trials_to_plot list:
        frames_to_plot = np.where(trial_numbers == trial)[0]
        # append frames to list of frames to plot (so as to get frames before the stim onset):
        frames_to_plot = np.concatenate((np.arange(frames_to_plot[0]-15, frames_to_plot[0]), frames_to_plot))
        # Get times and relative times for plotting on x axis for this trial:
        times_this_trial = camera_times[frames_to_plot]
        stim_on_this_trial = trials['stimOn_times'][trial-1]
        stim_right_this_trial = np.abs(trials['contrastRight'][trial - 1])
        stim_left_this_trial = np.abs(trials['contrastLeft'][trial - 1])
        if np.isnan(stim_right_this_trial):
            stim_right_this_trial = 0
        if np.isnan(stim_left_this_trial):
            stim_left_this_trial = 0
        stim_signed_contrast = (stim_right_this_trial - stim_left_this_trial)*100
        choice_this_trial = -trials['choice'][trial - 1]
        reward_this_trial = trials['feedbackType'][trial-1]
        reward_time_this_trial = trials['feedback_times'][trial-1] - stim_on_this_trial
        relative_times_this_trial = times_this_trial - stim_on_this_trial
        # Extract a single video frame and overlay location of top of pupil onto it
        for k, frame in enumerate(frames_to_plot):
            print("Plotting frame " + str(k) + " of " + str(len(frames_to_plot)) + " for trial " + str(trial) + '; see directory ' + str(dir_for_plotting))
            frame_time = relative_times_this_trial[k]
            video_frame = get_video_frame(str(raw_vid_meta), frame)
            # Overlay a circle corresponding to top of pupil onto image:
            pupil_xy = create_frame_array(dlc_dict, frame)[0]

            # Extract Guido's circle coordinates and rad:
            this_x_circ = x_circ_denoised[frame]
            this_y_circ = y_circ_denoised[frame]
            this_circ_rad = michael_d_denoised[frame]/ 2

            fig = plt.figure(figsize=(10, 15), dpi=80, facecolor='w', edgecolor='k')
            plt.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.90, wspace=0.3, hspace=0.2)

            ax = plt.subplot(3, 1, 1)
            plt.imshow(video_frame)
            for j in range(len(pupil_xy)):
                ax.scatter(pupil_xy[j][0], pupil_xy[j][1], marker='+', s=100)
            circle1 = plt.Circle((this_x_circ, this_y_circ), this_circ_rad, color='r', fill=False)
            ax.add_artist(circle1)
            # get session date
            match = re.search(r'\d{4}-\d{2}-\d{2}', str(save_path))
            sess_date = match.group()
            # get animal:
            match2 = re.search('Subjects/(.*)/', str(save_path))
            animal = match2.group(1)
            plt.title(animal + "; Trial " + str(int(trial)) +'\n Signed contrast = ' + str(stim_signed_contrast) + '; Rewarded = ' + str(reward_this_trial) + '\n Frame ' + str(k), fontsize=28)

            plt.subplot(3, 1, 2)
            plt.plot(relative_times_this_trial, michael_d[frames_to_plot], 'o-', label = 'raw data', alpha = 0.5)
            plt.plot(relative_times_this_trial, michael_d_denoised[frames_to_plot], 'o-', label='after wavelet transform', alpha=0.5)
            plt.axvline(x = 0, label = 'stim on', alpha = 0.5, color = 'g', linewidth =3)
            plt.axvline(x=max(relative_times_this_trial), label='stim on', alpha=0.5, color='g', linewidth=3)
            plt.axvline(x=reward_time_this_trial, label='feedback time', alpha=0.5, color = 'r', linewidth =3)
            plt.axvline(x=frame_time, alpha=0.5, color='k')
            plt.title("Diameter")
            plt.ylabel("diameter (px)")
            plt.xlabel("time relative to stimOn (seconds)")
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(relative_times_this_trial, dlc_dict['pupil_top_r_likelihood'][frames_to_plot],  label = 'top', alpha= 0.5)
            plt.plot(relative_times_this_trial, dlc_dict['pupil_bottom_r_likelihood'][frames_to_plot], label='bottom', alpha= 0.5)
            plt.plot(relative_times_this_trial, dlc_dict['pupil_right_r_likelihood'][frames_to_plot], label='right', alpha= 0.5)
            plt.plot(relative_times_this_trial, dlc_dict['pupil_left_r_likelihood'][frames_to_plot],  label='left', alpha= 0.5)
            plt.axvline(x=0, label='stim on', alpha=0.5,color = 'g', linewidth =3)
            plt.axvline(x=max(relative_times_this_trial), label='stim on', alpha=0.5, color='g', linewidth=3)
            plt.axvline(x=reward_time_this_trial, label='feedback time', alpha=0.5, color = 'r', linewidth =3)
            plt.axvline(x=frame_time, alpha=0.5, color='k')
            plt.legend()
            plt.ylim((-0.05, 1.05))
            plt.title("DLC likelihood")
            plt.ylabel("dlc likelihood")
            plt.xlabel("time relative to stimOn (seconds)")
            fig.savefig(dir_for_plotting + 'frame_' + "{0:03}".format(k) + '.png')