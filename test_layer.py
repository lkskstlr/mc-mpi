#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_me = pd.read_csv("build/WA.out", sep=" ", header=None, names=['x', 'weight'])
df_ref = pd.read_csv("src_test/test_layer_target_WA.out", sep=" ", header=None, names=['x', 'weight'])

plt.figure(figsize=(10,10))
plt.plot(df_me['x'], df_me['weight'], '+-', label="me")
plt.plot(df_ref['x'], df_ref['weight'], 'm--.', label="reference")
plt.title("Weights after Simulation")
plt.legend()
plt.savefig("fig_weights.png", format='png', dpi=300)