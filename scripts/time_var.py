#!/bin/env python3
import subprocess
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run_timings(nb_particles):
    p = subprocess.Popen("./test_layer_time_var {}".format(nb_particles), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
    times = np.empty((len(lines, )), dtype=np.float)
    times[:] = np.nan
    pattern = re.compile("^\d+.\d+$")
    for i, line in enumerate(lines):
        res = re.match(pattern, line)
        if res:
            times[i] = float(line)
             
    return times

def plot_fig(nb_particles, means, stds):
    
    plt.figure()
    normalizer = nb_particles[0]/means[0]
    plt.plot(nb_particles, normalizer*means/nb_particles, '+-', label='mean time')
    plt.fill_between(nb_particles, normalizer*(means+stds)/nb_particles, normalizer*(means-stds)/nb_particles, facecolor='blue', alpha=0.5, label="standard deviation")
    plt.legend()
    plt.xlabel("Number of particles")
    plt.ylabel("Time / Number of particles")
    plt.show()
    

if __name__ == "__main__":
    filename = "time_var.pickle"
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            nb_particles, means, stds = pickle.load(file)
    else:
        os.chdir("../build")
        nb_particles = np.array([10000, 50000, 100000, 500000, 1000000])
        means = np.zeros(nb_particles.shape, dtype=np.float)
        stds = np.zeros(nb_particles.shape, dtype=np.float)
        
        for i in range(len(nb_particles)):
            times = run_timings(nb_particles[i])
            means[i] = np.mean(times)
            stds[i] = np.std(times)
            
            print("{:7d}: {} +- {} %".format(nb_particles[i], means[i], 100*stds[i]/means[i]))
            
        with open("../scripts/"+filename, 'wb') as file:
            pickle.dump([nb_particles, means, stds], file)
        
    plot_fig(nb_particles, means, stds)