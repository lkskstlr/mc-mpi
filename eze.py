#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:09:36 2019

@author: ezequiel.morcillo-rosso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = dict()
data['df_w'] = pd.read_csv("build/weights.csv", skipinitialspace=True)
data['data_w'] = np.genfromtxt('tex/data/WA_1000_1000000.out', delimiter=' ')

df_w = data['df_w']
data_w = data['data_w']
plt.figure(figsize=(20,20))
plt.plot(df_w['x'], df_w['weight'], '+-', label="parallel")
plt.plot(data_w[:, 0], data_w[:, 1], 'm--.', label="sequential (reference)")
plt.title("Weights after Simulation")
plt.legend()
plt.savefig("fig_weights.png", format='png', dpi=300)