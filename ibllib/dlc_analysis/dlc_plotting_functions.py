# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:55:33 2019

List of plotting functions for DLC data

@author: Guido, Kelly
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore


def peri_plot(trace, timestamps, events, ax, window_centers, window_size, norm='none'):
    """
    Plot a peri-plot centered around a behavioral event
    :trace:        1D array with trace to be plotted
    :timestamps:   1D array with timestamps in seconds
    :events:       1D array with event times to center around
    :ax:           Axes to plot in, set to None to skip plotting
    :time_window:  time window in seconds
    :norm:         how to perform normalization
                   'none': no normalization (default)
                   'zscore': z-score the entire trace
                   'baseline': subtract the baseline from each trial trace
                               defined as the time before onset of the event
    """

    # Transform time window into samples
  #  sampling_rate = 1 / np.mean(np.diff(timestamps))
  #  sample_win = [np.int(np.round(time_win[0] * sampling_rate)),
   #               np.int(np.round(time_win[1] * sampling_rate))]

    # Z-score entire trace
    if norm == 'zscore':
        trace = zscore(trace)

    # Create dataframe for line plot
    peri_df = pd.DataFrame(columns=['event_nr', 'timepoint', 'trace'])
    for i in np.arange(np.size(events)):

        # Get trace for this trial
        this_time = timestamps-events[i]
        this_trace = np.zeros(0)
        for j in range(len(window_centers)):
            this_trace = np.append(this_trace, np.mean(
                    trace[((this_time > window_centers[j]-(window_size/2))
                           & (this_time < window_centers[j]+(window_size/2)))]))

        # Perform baseline correction
        if norm == 'baseline':
            this_trace = this_trace - np.median(
                    this_trace[window_centers < window_centers[0]/2])

        # Add to dataframe
        this_df = pd.DataFrame(data={'event_nr': np.ones(np.size(this_trace),
                                                         dtype=int)*(i+1),
                                     'timepoint': window_centers, 'trace': this_trace})
        peri_df = pd.concat([peri_df, this_df], ignore_index=True)
    return peri_df

    # Plot
    if ax is not None:
        sns.lineplot(x='timepoint', y='trace', data=peri_df, ci=68, ax=ax)
