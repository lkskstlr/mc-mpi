#!/usr/bin/env python3
import subprocess
import yaml
import pickle
import os
import re
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

filename = "../intra-node.pkl"
foldername = "intra-node"
m = 1


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
    
    ns = np.arange(1, 9)
    modes = ("sync", "rma", "async")
    comms = ("mpi", "openmp")
    
    times = dict()
    
    for comm in comms:
        for mode in modes:
            times[comm+"+"+mode] = np.empty((ns.size, ), dtype=np.float)
            for i in range(ns.size):
                _d = [d['time'] for d in data.values() if (d['mode'] == mode and d['n'] == ns[i] and d['type'] == comm)]
                assert len(_d) == m, "Found not exactly {} time".format(m)
                times[comm+"+"+mode][i] = np.median(np.array(_d))
    
    plt.figure()
    for comm in comms:
        for mode in modes:
            plt.plot(ns, times[comm+"+"+mode], '+-', label=comm+"+"+mode)
    plt.legend()
    plt.show()
    
    t_ref = 27.0
    rs = dict()
    for comm in comms:
        for mode in modes:
            rs[comm+"+"+mode] = (t_ref / times[comm+"+"+mode]) / (ns/1)
    
    plt.figure()
    for comm in comms:
        for mode in modes:
            plt.plot(ns, rs[comm+"+"+mode], '+-', label=comm+"+"+mode)
    plt.legend()
    plt.show()
            

if __name__ == "__main__":
    os.chdir("../{}".format(foldername))
    
    data = load_data()
    ids = sorted([x for x in data.keys()])
    pprint(data)
    pprint(ids)
    times = get_times(ids)
    
    for i, time in zip(ids, times):
        if not np.isnan(time):
            if 'time' not in data[i]:
                data[i]['time'] = time
    
    plot_data(data)
    save_data(data)
    
    
    
    
    
    
    
