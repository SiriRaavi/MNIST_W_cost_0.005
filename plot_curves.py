#!/usr/bin/env python2

import matplotlib.pyplot as plt
import h5py
import numpy as np

import matplotlib as mpl
mpl.use('pdf')
width = 3.4
height = width / 1.4

# %% loading the HDF5 file
h5f = h5py.File('./saved_models/Results.h5', 'r')
all_acc = h5f['all_acc'][:]
loss = h5f['loss'][:]
h5f.close()

loss = loss[:500000]
all_acc = all_acc[:500000]


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

plt.figure()
plt.plot(np.linspace(0, 550000, 39), movingaverage(loss, 500)[0:-1:10000],
         'o', color='black', label='Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')

acc_av = np.zeros_like(all_acc)
for i in range(all_acc.shape[-1]):
    acc_av[:, i] = movingaverage(all_acc[:, i], 500)

label = ['$1^{st}$', '$2^{nd}$', '$3^{rd}$', '$4^{th}$', '$5^{th}$', '$6^{th}$',
         '$7^{th}$', '$8^{th}$', '$9^{th}$', '$10^{th}$']

c = ['lightcoral', 'darkorange', 'royalblue', 'darkorchid', 'olive',
     'indigo', 'firebrick', 'saddlebrown', 'mediumslateblue', 'darkblue']

fig = plt.figure()
for i in [0, 1, 4, 9]:
    line, = plt.plot(np.linspace(0, 550000, 39), acc_av[1:-1:10000, i], 'o',
             color=c[i], label= label[i]+' instance')
plt.ylim([0, 1.0])
plt.xlabel('Episode')
plt.ylabel('Percent Correct')
plt.legend(loc=4)
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=15)
plt.show()

