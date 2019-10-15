# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:08:01 2019

List of analysis functions for DLC

@author: guido, miles
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt, stft
import sys
sys.path.insert(0, '/home/guido/Projects/ibllib/ibllib/dlc_analysis')
from dlc_basis_functions import px_to_mm


def butter_filter(data, cutoff, fs, ftype='lowpass', order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=ftype, analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def lick_times(dlc_dict, lick_threshold=1, bout_threshold=0.5):
    """
    Lick onset and offset frames.  If more than one dict is provided licks detected
    by one or the other are counted.
    :param dlc_data: tuple of tongue x and y positions.
    If provided only contact with tube is counted as a lick.

    :return: tuple of lick onset and offset frames
    """

    # Check if DLC output is in mm
    if ('units' not in dlc_dict.keys()) or (dlc_dict['units'] == 'px'):
        dlc_dict = px_to_mm(dlc_dict)

    # Get distances of tongue points to tube points
    tongue_x = np.mean([dlc_dict['tongue_end_l_x'], dlc_dict['tongue_end_r_x']], axis=0)
    tongue_y = np.mean([dlc_dict['tongue_end_l_y'], dlc_dict['tongue_end_r_y']], axis=0)
    tube_x = np.mean([dlc_dict['tube_top_x'], dlc_dict['tube_bottom_x']], axis=0)
    tube_y = np.mean([dlc_dict['tube_top_y'], dlc_dict['tube_bottom_y']], axis=0)
    dist = np.sqrt(((tongue_x - tube_x)**2) + ((tongue_y - tube_y)**2))

    # Get lick times
    all_lick = dist < lick_threshold
    licks = []
    for l in [i for i, x in enumerate(all_lick) if x]:
        if all_lick[l-1] == 0:
            licks = licks + [l]
    lick_times = dlc_dict['times'][licks]

    # Calculate power spectral densitity to get licking bouts
    sampling_rate = 1/np.mean(np.diff(dlc_dict['times']))
    freqs, time_filt, spec = signal.spectrogram(dist, fs=sampling_rate,
                                                nperseg=int(1*sampling_rate),
                                                noverlap=int(0.25*sampling_rate))

    # Define licking bouts as times at which the spectral density of frequencies between 6 and 12Hz
    # are higher than the bout_threshold
    power = np.mean(spec[(freqs > 6) & (freqs < 8)], axis=0)
    licking_bout_times = time_filt[power > np.std(power)*bout_threshold]
    licking_bout_times = licking_bout_times + dlc_dict['times'][0]

    return lick_times, licking_bout_times, dist


def blink_times(dlc_dict, threshold=0.9):
    pupil_likelihood = np.mean([dlc_dict['pupil_top_r_likelihood'],
                                dlc_dict['pupil_left_r_likelihood'],
                                dlc_dict['pupil_right_r_likelihood'],
                                dlc_dict['pupil_bottom_r_likelihood']], axis=0)
    all_blink = pupil_likelihood < threshold
    blinks = []
    for l in [i for i, x in enumerate(all_blink) if x]:
        if all_blink[l-1] == 0:
            blinks = blinks + [l]
    blink_times = dlc_dict['times'][blinks]
    return blink_times, pupil_likelihood


def sniff_times(dlc_dict, threshold=0.5):
    '''
    Sniff onset times.
    :param dlc_dict: Dict containing the following keys:
                  nostril_top_x, nostril_top_y, nostril_bottom_x, nostril_bottom_y
    :return: 1D array of sniff onset times
    '''

    # Check if DLC output is in mm
    if ('units' not in dlc_dict.keys()) or (dlc_dict['units'] == 'px'):
        dlc_dict = px_to_mm(dlc_dict)

    # Calculate distance between nostrils
    dist = np.sqrt(((dlc_dict['nostril_top_x'] - dlc_dict['nostril_bottom_x'])**2)
                   + ((dlc_dict['nostril_top_y'] - dlc_dict['nostril_bottom_y'])**2))

    # Calculate power spectral densitity to get licking bouts
    sampling_rate = 1/np.mean(np.diff(dlc_dict['times']))
    freqs, time_filt, spec = signal.spectrogram(dist, fs=sampling_rate,
                                                nperseg=int(1*sampling_rate),
                                                noverlap=int(0.5*sampling_rate))

    # Define licking bouts as times at which the spectral density of frequencies between 6 and 12Hz
    # are higher than the bout_threshold
    power = np.mean(spec[(freqs > 7) & (freqs < 12)], axis=0)
    sniffing_times = time_filt[power > np.std(power)*threshold]
    sniffing_times = sniffing_times + dlc_dict['times'][0]
    return sniffing_times


def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)
    return xc_1, yc_1, R_1


def pupil_features(dlc_dict):
    vec_x = [dlc_dict['pupil_left_r_x'], dlc_dict['pupil_right_r_x'],
             dlc_dict['pupil_top_r_x'], dlc_dict['pupil_bottom_r_x']]
    vec_y = [dlc_dict['pupil_left_r_y'], dlc_dict['pupil_right_r_y'],
             dlc_dict['pupil_top_r_y'], dlc_dict['pupil_bottom_r_y']]
    x = np.zeros(np.size(vec_x[0]))
    y = np.zeros(np.size(vec_x[0]))
    diameter = np.zeros(np.size(vec_x[0]))
    for i in range(np.size(vec_x[0])):
        try:
            x[i], y[i], R = fit_circle([vec_x[0][i], vec_x[1][i], vec_x[2][i]],
                                       [vec_y[0][i], vec_y[1][i], vec_y[2][i]])
            diameter[i] = R*2
        except:
            x[i] = np.nan
            y[i] = np.nan
            diameter = np.nan

    return x, y, diameter
