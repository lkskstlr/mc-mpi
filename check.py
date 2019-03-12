#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = dict()
data['df_w'] = pd.read_csv("build/out/weights.csv", skipinitialspace=True)
data['data_w'] = np.genfromtxt('tex/data/WA_1000_1000000.out', delimiter=' ')

df_w = data['df_w']
data_w = data['data_w']
plt.figure(figsize=(10,10))
plt.plot(df_w['x'], df_w['weight'], '+-', label="parallel")
plt.plot(data_w[:, 0], data_w[:, 1], 'm--.', label="sequential (reference)")
plt.title("Weights after Simulation")
plt.legend()
plt.show()

stats = pd.read_csv("build/out/stats.csv", skipinitialspace=True)
stats = stats.loc[:, ~stats.columns.str.contains('^Unnamed')]
time_min = stats['starttime'].min()
stats['starttime'] -= time_min
stats['endtime'] -= time_min

world_size = stats['rank'].max()+1
plt.figure(figsize=(9,8/5*world_size))
for proc in range(0, world_size):
    X = stats[stats['rank'] == proc]
    if proc == 0:
        ax0 = plt.subplot(world_size, 1, proc+1)
        ax = ax0
    else:
        ax = plt.subplot(world_size, 1, proc+1, sharex=ax0)
    
    plt.stackplot(X['starttime'].values,
        X['time_comp'].values,
        X['time_send'].values,
        X['time_recv'].values,
        labels=["Comp", "MPI_Sendrecv", "MPI_Allreduce"])
    
        
    
    
    if proc == world_size-1:
        plt.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("Seconds")
        plt.ylabel("Seconds")
    else:
        plt.ylabel(str(proc))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        #ax.set_xticklabels([])
    
plt.tight_layout()    
plt.show()
        
f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))
x1 = np.arange(0., 0.707, 0.001)
x2 = np.arange(0.708, 1.0, 0.001)
# plot the same data on both axes
ax.plot([0.706, 0.707, 0.708], [70.959, 2100, 70.959], 'b', linewidth = 2.2)
ax2.plot(x1, 41*x1 + 42.013, 'b', x2, 41 * -x2 + 99.987, 'b', [0.706, 0.707, 0.708], [70.959, 2100, 70.959], 'b')

# zoom-in / limit the view to different portions of the data
ax.set_ylim(2050, 2150)  # outliers only
ax2.set_ylim(35, 90)  # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


plt.ylabel('Weight')
plt.xlabel('x')

plt.tight_layout()
plt.show()

for proc in range(0, world_size):
    X = stats[stats['rank'] == proc]
    print(np.sum(X['time_comp'].values))    