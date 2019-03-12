#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

if __name__ == "__main__":
    nthreads = np.arange(1, 9)
    
    if False:
        times = 100*np.ones(nthreads.shape, dtype=np.float)
        
        for _ in range(1):
            for i, nthread in enumerate(nthreads):
                p = subprocess.Popen("build/test_layer_perf {}".format(nthread), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                p.wait()
                lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
                times[i] = min(float(lines[0]), times[i])
                print(times[i])
    else:  

        times_static = np.load("times_static_3.npy")
        times_dynamic = np.load("times_dynamic_1.npy")  
        
        fig = plt.figure(figsize=(4,4))
        plt.plot(nthreads, times_static[0]/times_static, '-o', label="static")
        plt.plot(nthreads, times_dynamic[0]/times_dynamic, '-o', label="dynamic")
        plt.plot(nthreads[0:4], nthreads[0:4], 'r:', label='optimal')
        plt.xlabel("OpenMP threads")
        plt.ylabel("Speedup")
        plt.title("OpenMP speedup")
        plt.grid("on")
        plt.legend()
        plt.show()
        fig.savefig("openmp_speedup.png", dpi=300, bbox_inches='tight', pad_inches=0)
        