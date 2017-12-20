# -*- coding: utf-8 -*-
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
#
# Modified by 
#   Giulio Degano
#   PhD Student
#   Computational Cognitive Neuroimaging Laboratory
#   School of Psychology
#   College of Life and Environmental Sciences
#   University of Birmingham
#   Edgbaston
#   Birmingham B15 2TT
#   e-mail: gxd606@student.bham.ac.uk

# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem_fname = (data_path +
             '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')

# Load real data as the template
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference(projection=True)
raw = raw.crop(0., 30.)  # 30 sec is enough

# Generation of two dipoles abd the data

n_dipoles = 4  # number of dipoles to create
epoch_duration = 2.  # duration of each epoch/event
n = 0  # harmonic number


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz"""
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 25e-9 * np.sin(2. * np.pi * 10. * n * times)
    data *= window
    return data


times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
src = read_source_spaces(src_fname)
stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                          data_fun=data_fun, random_state=0)

# Simulate data with given template
raw_sim = simulate_raw(raw, stc, trans_fname, src, bem_fname, cov='simple',
                       iir_filter=[0.2, -0.2, 0.04], ecg=True, blink=True,
                       n_jobs=1, verbose=True)

# Epoching

events = find_events(raw_sim)  # only 1 pos, so event number == 1
epochs = Epochs(raw_sim, events, 1, -0.4, epoch_duration,preload=True)
# plot
epochs.plot()

# Prestim extraction 
prestim=epochs.crop(tmin=-0.4, tmax=0)
prestim_avg=prestim.average

# Now we have a Data_frame.... fun begins
df=prestim_avg.to_data_frame()







