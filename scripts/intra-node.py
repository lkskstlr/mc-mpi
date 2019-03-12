#!/bin/env python3
import subprocess
import yaml
import pickle
import os
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

filename = "intra-node.pkl"
foldername = "intra-node"
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
        
def mpirun(n, mode, test=True):
    if test:
        print(n)
        print(mode)
        with open("py_config.yaml", 'r') as file:
            pprint(yaml.load(file))
        return None
    
    p = subprocess.Popen("mpirun -n {} ./main py_config.yaml {}".format(n, mode), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
    return float(lines[0])
    


if __name__ == "__main__":
    os.chdir("../{}".format(foldername))
    
    nb_particles = int(1e6)
    ns = np.arange(1, 9)
    modes = ("sync", "rma", "async")
    
    data = load_data()
    pprint(data)
    
    if False:
        for mode in modes:
            data[mode] = np.zeros(ns.shape, dtype=np.float)
            for i,n in enumerate(ns):
                    write_config(mode=mode, nb_particles=nb_particles, nthread=1)
                    _time = mpirun(n, mode, test=test)
                    
                    if _time is not None:
                        data[mode][i] = _time
                        print("n = {}, {}, {}".format(n, mode, _time))
                        save_data(data)
                        
    else:
        times_omp = np.load("../times_static_3.npy")
        fig = plt.figure(figsize=(5,3))
        for mode in modes:
            plt.plot(ns, data[mode][0]/data[mode], '+-', label=mode)
        plt.plot(ns, times_omp[0]/times_omp, '+-', label="OpenMP")
        plt.plot(ns[0:4], ns[0:4], 'r:', label='optimal')
        plt.xlabel("processes/threads")
        plt.ylabel("Speedup")
        plt.title("MPI/OpenMP for shared memory")
        plt.grid("on")
        plt.legend()
        plt.show()
        fig.savefig("shared_memory.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
    
    
