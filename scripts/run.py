#!/usr/local/anaconda3/bin/python3
import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt

def main_omp():
    my_env = os.environ.copy()
    my_env['OMP_NUM_THREADS'] = "2"
    p = subprocess.Popen("salloc -n 5 -N 5 mpirun ./main ../config.yaml sync", env=my_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    retval = p.wait()

def layer_perf():
    m = 2
    nthreads = np.arange(1, 9)
    means = np.zeros(nthreads.shape, dtype=np.float)
    std = np.zeros(nthreads.shape, dtype=np.float)
    times = np.zeros((m,), dtype=np.float)

    for i, nthread in enumerate(nthreads):
        print(nthread, end=": ")
        for j in range(m):
            p = subprocess.Popen("./test_layer_perf {:d}".format(nthread), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = [x.decode('ascii').rstrip() for x in p.stdout.readlines()]
            retval = p.wait()
            times[j] = float(lines[0])
            print(times[j], end=" ")
        
        print("")
        means[i] = np.mean(times)
        std[i] = np.std(times)

    return nthreads, means, std


def main_scaling():
    m = 1
    modes = ["sync", "async", "rma"]
    N = np.array([1, 2, 4, 5, 8, 10])
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
    res = main_scaling()
    
    for mode in res['modes']:
        plt.plot(res['N'], res[mode]['means'], '+-', label=mode)
    plt.legend()
    plt.show()