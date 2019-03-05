#!/bin/env python3
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
import os


def layer_perf():
    m = 2
    nthreads = np.arange(1, 9)
    # nthreads = np.array([4, 8])
    means = np.zeros(nthreads.shape, dtype=np.float)
    std = np.zeros(nthreads.shape, dtype=np.float)
    times = np.zeros((m,), dtype=np.float)

    for i, nthread in enumerate(nthreads):
        print(nthread, end=": ")
        for j in range(m):
            p = subprocess.Popen("./test_layer_perf {:d}".format(nthread), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
            p.wait()
            times[j] = float(lines[0])
            print(times[j], end=" ")
        
        print("")
        means[i] = np.mean(times)
        std[i] = np.std(times)
        
    return nthreads, means, std


def main_scaling():
    m = 2
    modes = ["sync", "async", "rma"]
    N = np.array([1, 2, 4, 5, 8, 10])
    #N = np.array([1, 5])
    res = dict()
    res["N"] = N
    res['modes'] = modes

    for mode in modes:
        print(mode)
        res[mode] = dict()
        res[mode]["means"] = np.zeros(N.shape, dtype=np.float)
        res[mode]["std"]   = np.zeros(N.shape, dtype=np.float)
        for i, n in enumerate(N):
            times = np.zeros((m,), dtype=np.float)
            print("\t{}: ".format(n), end="")
            for j in range(m):
                s = "salloc -N {:d} -n {:d} mpirun ./main ../config.yaml {}".format(n, n, mode)
                p = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                p.wait()
                lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
                times[j] = float(lines[1])
                print(times[j], end=" ")
                
            res[mode]["means"][i] = np.mean(times)
            res[mode]["std"][i] = np.std(times)
            print("")
            
    

    return res


if __name__ == "__main__":
    if not os.path.exists("data.pickle"):
        print("Running Simulations")
        nthreads, means, std = layer_perf()
        res = main_scaling()
        
        data = {
            'nthreads': nthreads,
            'means': means,
            'res': res
        }
        
        with open('data.pickle', 'wb') as file:
            pickle.dump(data, file)

    with open('data.pickle', 'rb') as file:
        print("Loading data")
        data = pickle.load(file)


    plt.figure()
    plt.plot(data['nthreads'], data['means'], '+-')
    plt.title("OpenMP Scaling Analysis")
    plt.xlabel("Number of OMP Threads")
    plt.ylabel("Runtime in s")
    plt.savefig("omp.png")

    plt.figure()
    for mode in data['res']['modes']:
        plt.plot(data['res']['N'], data['res'][mode]['means'], '+-', label=mode)
    plt.legend()
    plt.title("MPI Scaling Analysis")
    plt.xlabel("Number of MPI nodes")
    plt.ylabel("Runtime in s")
    plt.savefig("scaling.png")
    
    
