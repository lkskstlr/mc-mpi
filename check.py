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
        X['time_idle'].values,
        labels=["Comp", "Send", "Recv", "Idle"])
    
        
    
    
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
        ax.set_xticklabels([])
        
#plt.figure(figsize=(10,10))
plt.tight_layout()
plt.show()

for proc in range(0, world_size):
    X = stats[stats['rank'] == proc]
    print(np.sum(X['time_comp'].values))    