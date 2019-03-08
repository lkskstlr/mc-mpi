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

sha = "fc03906c5ae795e1450254408bee710ac1deee78"
filename = "../py_data.pkl"
foldername = "experiment01"
test = False
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
        
def write_batch(N, n, mode):
    lines = list()
    lines.append("#!/bin/bash")
    if N is not None:
        lines.append("#SBATCH -N {}".format(N))
    if n is not None:
        lines.append("#SBATCH -n {}".format(n))
        
    modes = ("sync", "rma", "async")
    if not (mode in modes):
        raise ValueError("Mode must be in {}".format(modes))
    lines.append("mpirun ./main py_config.yaml {}".format(mode))
    
    with open("py_run.batch", "w") as file:
        for line in lines:
            print(line, file=file)
            
def write_config(mode, nb_particles = 1000000, nthread=-1):
    nb_particles_per_cycle = {
        'sync': 100000,
        'async': 500,
        'rma': 65000    
    }
    
    config = {
            'x_min': 0.0,
            'x_max': 1.0,
            'x_ini': 0.7071067690849304,
            'nb_cells': 1000,
            'nb_particles': int(nb_particles),
            'particle_min_weight': 1e-12,
            'cycle_time': 1e-5,
            'statistics_cycle_time': 0.1,
            'nthread': nthread
            }
    modes = ("sync", "rma", "async")
    if not (mode in modes):
        raise ValueError("Mode must be in {}".format(modes))
    config['nb_particles_per_cycle'] = nb_particles_per_cycle[mode]
    
    with open("py_config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)
        
def sbatch(test=True):
    if test:
        with open("py_run.batch", 'r') as file:
            print(file.read())
        with open("py_config.yaml", 'r') as file:
            pprint(yaml.load(file))
        return None
    
    p = subprocess.Popen("sbatch py_run.batch", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
    for line in lines:
        print(line)
    exp = re.compile("Submitted batch job (\d+)")
    res = re.match(exp, lines[0])
    if res is None:
        raise ValueError("Could not match job")
    return int(res.group(1))
    
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
            _d = [d['time'] for d in data.values() if (d['sha'] == sha and d['mode'] == mode and d['n'] == ns[i] and d['N'] == Ns[i])]
            assert len(_d) == m, "Found not exactly {} time".format(m)
            times[mode][i] = np.median(np.array(_d))
    
    plt.figure()
    for mode in modes:
        plt.plot(ns, times[mode], '+-', label=mode)
    plt.legend()
    plt.show()
    
    t_ref = 2700.0
    rs = dict()
    for mode in modes:
        rs[mode] = (t_ref / times[mode]) / (ns/1)
    
    plt.figure()
    for mode in modes:
        plt.semilogy(ns, rs[mode], '+-', label=mode)
    plt.legend()
    plt.show()
            

if __name__ == "__main__":
    os.chdir("../{}".format(foldername))
    
    data = load_data()
    ids = sorted([x for x in data.keys() if (data[x]['foldername'] == foldername and data[x]['sha'] == sha)])
    pprint(data)
    pprint(ids)
    times = get_times(ids)
    
    for i, time in zip(ids, times):
        if not np.isnan(time):
            if 'time' not in data[i]:
                data[i]['time'] = time
    
    plot_data(data)
    save_data(data)
    
    
    
    
    
    
