#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:29:29 2019

@author: lukas.koestler
"""

import subprocess
import yaml
import pickle
import os
import re
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt


sha = "2a50a4c96ee899af3e325fedd5537bd2c3a14d1a"
filename = "../py_data.pkl"
foldername = "experiment03"
m = 5

def load_data():
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
            print("Loaded data with {} entries".format(len(data)))
            return data
    print("New data")
    data = {}
    return data

def save_data(data):
    print("Save data")
    with open(filename, "wb") as file:
        pickle.dump(data, file)
        
    
def get_times(ids):
    times = np.empty((len(ids, )), dtype=np.float)
    times[:] = np.nan
    pattern = re.compile("^\d+.\d+$")
    for index, i in enumerate(ids):
        if os.path.exists("slurm-{}.out".format(i)):
            with open("slurm-{}.out".format(i), 'r') as file:
                lines = [x.rstrip() for x in file.readlines()]
            for line in lines:
                res = re.match(pattern, line)
                if res:
                    times[index] = float(line)
    return times

def extract_data(data, sha):
    new_data = dict()
    for job_id in data.keys():
        if data[job_id].get('sha') == sha:
            new_data[job_id] = data[job_id].copy()
    return new_data

def plot_data(data):
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 80])
    Ns = np.ceil(ns/8).astype(np.int)
    modes = ("sync", "rma", "async")
    times = dict()
    
    for mode in modes:
        times[mode] = np.empty((ns.size, ), dtype=np.float)
        times[mode][:] = np.nan
        for i in range(ns.size):
            _d = [d['time'] for d in data.values() if (d['foldername'] == foldername and d['sha'] == sha and d['mode'] == mode and d['n'] == ns[i] and d['N'] == Ns[i])]
            assert len(_d) == m, "Found not exactly {} time".format(m)
            times[mode][i] = np.min(np.array(_d))
    
    fig = plt.figure(figsize=(2,2))
    for mode in modes:
        plt.plot(ns, times[mode], '+-', label=mode)
    plt.legend()
    plt.xlabel("Num. MPI processes")
    plt.ylabel("Runtime [s]")
    plt.title("Runtime for pure MPI")
    plt.grid("on")
    plt.show()
    fig.savefig("pure_mpi_runtime.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    
    t_ref = 27.0
    rs = dict()
    for mode in modes:
        rs[mode] = (t_ref / times[mode]) / (ns/1)
    
    fig = plt.figure(figsize=(2,2))
    for mode in modes:
        plt.semilogy(ns, rs[mode], '+-', label=mode)
    plt.legend()
    plt.xlabel("Num. MPI processes")
    plt.ylabel("Parallel Efficiency")
    plt.title("Parallel efficiency for pure MPI")
    plt.grid("on")
    plt.show()
    fig.savefig("pure_mpi_efficiency.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    fig = plt.figure(figsize=(2,2))
    for mode in modes:
        plt.plot(ns, rs[mode], '+-', label=mode)
    plt.legend()
    plt.xlabel("Num. MPI processes")
    plt.ylabel("Parallel Efficiency")
    plt.title("Parallel efficiency for pure MPI")
    plt.grid("on")
    plt.show()
    fig.savefig("pure_mpi_efficiency_log.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
            

if __name__ == "__main__":
    os.chdir("../{}".format(foldername))
    
    data = load_data()
    ids = sorted([x for x in data.keys() if (data[x]['foldername'] == foldername and data[x]['sha'] == sha)])
    times = get_times(ids)
    times[np.isnan(times)] = 10*np.nanmax(times)
    
    for i, time in zip(ids, times):
        if not np.isnan(time):
            if 'time' not in data[i]:
                data[i]['time'] = time
    
    plot_data(data)
    save_data(data)
    
    
    
    
    
    
