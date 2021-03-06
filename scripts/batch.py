#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:48:36 2019

@author: lukas.koestler
"""

#!/bin/env python3
import subprocess
import yaml
import pickle
import os
import re
from pprint import pprint
import numpy as np

sha = "fc03906c5ae795e1450254408bee710ac1deee78"
filename = "../py_data.pkl"
foldername = "experiment01"
test = False

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
    

def check_env():
    p = subprocess.Popen("which mpirun", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
    assert lines[0] == "/users/profs/2017/francois.trahay/soft/install/openmpi/bin/mpirun", "Not running correct environment. source set_env.sh?"


if __name__ == "__main__":
    os.chdir("../{}".format(foldername))
    check_env()
    
    data = load_data()
    pprint(data)
    
    nthread = 1
    nb_particles = int(1e8)
    
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 80])
    Ns = np.ceil(ns/8).astype(np.int)
    modes = ("sync", "rma", "async")
    
    
    for n, N in zip(ns, Ns):
        for mode in modes:
            write_config(mode=mode, nb_particles=nb_particles, nthread=nthread)
            write_batch(N=N, n=n, mode=mode)
    
            job_id = sbatch(test=test)
            if job_id is not None:
                data[job_id] = {'N': N, 'n': n, 'mode': mode, 'nb_particles': nb_particles, 'foldername': foldername, 'sha': sha}
            save_data(data)
    
            
    save_data(data)
    
    
    
