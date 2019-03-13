#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


slurm_job_id = 38455
folderpath = "out/{}/".format(slurm_job_id)

def decompose_domain2(world_size, h, m, c):
    z = 1.0/np.sqrt(2)
    I = h-0.5*m*(z**2 + (1-z)**2)+c
    print(I)
    
    dI = I / world_size
    x = np.zeros((world_size+1, ), dtype=np.float)
    for proc in range(world_size):
        x_min = x[proc]
        if x_min < z:
            print(x_min)
            coeff = np.array([0.5*m, h-m*z, -x_min*h + x_min*m*z -x_min**2*m/2 - dI])
            sols = np.roots(coeff)
            assert np.prod(sols) < 0, "Should have different signs {}".format(sols)
            x_max = sols[1]
            
            if x_max > z:
                print("reached peak")
                x_max -= np.sqrt(2*c/m)
                if x_max < z:
                    x_max = z+1e-2
        else:
            coeff = np.array([-0.5*m, h+m*z, -x_min*h - x_min*m*z + x_min**2*m/2 - dI])
            sols = np.roots(coeff)
            ind = np.where(np.logical_and(sols > x_min, sols <= 1.0))[0]
            assert len(ind) == 1
            x_max = sols[ind[0]]
        
        x[proc+1] = x_max
    return x

def decomp_linear(size, f0, m, dI):
    xs = np.zeros((size+1, ), dtype=np.float)
    for i in range(size):
        coeff = np.array([m/2, f0, -dI])
        sols = np.roots(coeff)
        ind = np.where(np.logical_and(sols >= 0, sols <=1))[0]
        assert len(ind) == 1, "sols = {}".format(sols)
        xs[i+1] = xs[i] + sols[ind[0]]
        f0 += m*sols[ind[0]]
    return xs

def decompose_domain(world_size, h, m, c):
    z = 1.0/np.sqrt(2)
    I = h-0.5*m*(z**2 + (1-z)**2)+c    
    dI = I / world_size
    coeff = np.array([-m/4, h, c-dI])
    sols = np.roots(coeff)
    ind = np.where(np.logical_and(sols > 0.0, sols <= 1.0))[0]
    assert len(ind) == 1
    x_min = z - 0.5*sols[ind[0]]
    x_max = z + 0.5*sols[ind[0]]
    
    f0left = h-z*m
    f0right = h - (x_max-z)*m
    Ileft = f0left*x_min + 0.5*x_min*x_min*m
    Iright = f0right*(1-x_max) - 0.5*(1-x_max)**2*m
    
    nleft = int(np.round((Ileft)/(Ileft+Iright) * (world_size-1)))
    nright = world_size- 1 - nleft
    
    xs_left = decomp_linear(nleft, f0left, m, Ileft/nleft)
    xs_right = decomp_linear(nright, f0right, -m, Iright/nright) + x_max
    
    
    assert np.allclose(xs_left[-1], x_min)
    assert np.allclose(xs_right[-1], 1.0)
    
    xs_left[-1] = x_min
    xs_right[-1] = 1.0
    
    return np.concatenate((xs_left, xs_right))
try:
    data = dict()
    data['df_w'] = pd.read_csv(folderpath+"weights.csv", skipinitialspace=True)
    data['data_w'] = np.genfromtxt('data/WA_1000_1000000.out', delimiter=' ')
    
    df_w = data['df_w']
    data_w = data['data_w']
    plt.figure(figsize=(10,10))
    plt.plot(df_w['x'], df_w['weight'], '+-', label="parallel")
    plt.plot(data_w[:, 0], data_w[:, 1], 'm--.', label="sequential (reference)")
    plt.title("Weights after Simulation")
    plt.legend()
    plt.show()
except:
    print("Couldn't do weights")

stats = pd.read_csv(folderpath+"stats.csv", skipinitialspace=True)
stats = stats.loc[:, ~stats.columns.str.contains('^Unnamed')]
time_min = stats['starttime'].min()
stats['starttime'] -= time_min
stats['endtime'] -= time_min

world_size = stats['rank'].max()+1
total_work = np.zeros((world_size,), dtype=np.float)
plt.figure(figsize=(9,8/5*world_size))
for proc in range(0, world_size):
    X = stats[stats['rank'] == proc]
    total_work[proc] = X['time_comp'].values.sum()
    if proc == 0:
        ax0 = plt.subplot(world_size, 1, proc+1)
        ax = ax0
    else:
        ax = plt.subplot(world_size, 1, proc+1, sharex=ax0)
    
    plt.stackplot(X['starttime'].values,
        X['time_comp'].values,
        X['time_send'].values,
        X['time_recv'].values,
        X['time_idle'].values,
        labels=["Comp", "Send", "Recv", "Idle"])
    
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if proc == world_size-1:
        plt.legend()
        plt.xlabel("Seconds")
        plt.ylabel("Seconds")
    else:
        plt.ylabel("Seconds")
        ax.get_xaxis().set_visible(False)
        
plt.show()

print(total_work)

c1 = np.zeros((world_size,), dtype=np.float)
c2 = np.zeros((world_size,), dtype=np.float)

for proc in range(0, world_size):
    x_min = 1.0/world_size*proc
    x_max = 1.0/world_size*(proc+1)
    dx = x_max-x_min
    
    if x_max < 1/np.sqrt(2):
        c1[proc] = dx
        c2[proc] = -dx/np.sqrt(2) + (x_max**2-x_min**2)/2
    elif x_min > 1/np.sqrt(2):
        c1[proc] = dx
        c2[proc] = +dx/np.sqrt(2) - (x_max**2-x_min**2)/2
    else:
        special = proc

_c1 = np.delete(c1, special)
_c2 = np.delete(c2, special)

A = np.hstack((_c1[:, None], _c2[:, None]))
b = np.delete(total_work, special)
sol = np.linalg.lstsq(A, b)[0]

xx = np.linspace(0, 1, 1000)
yy = b[0] - b[1]*(1/np.sqrt(2)-xx)
yy[xx>1/np.sqrt(2)] = b[0] - b[1]*(xx[xx>1/np.sqrt(2)] - 1/np.sqrt(2))


expectedwork = np.zeros((world_size,), dtype=np.float)

for proc in range(0, world_size):
    x_min = 1.0/world_size*proc
    x_max = 1.0/world_size*(proc+1)
    dx = x_max-x_min
    z = 1.0/np.sqrt(2)
    
    if x_max < 1/np.sqrt(2):
        expectedwork[proc] = dx*sol[0] + sol[1]*(-dx/np.sqrt(2) + (x_max**2-x_min**2)/2)
    elif x_min > 1/np.sqrt(2):
        expectedwork[proc] = dx*sol[0] + sol[1]*(+dx/np.sqrt(2) - (x_max**2-x_min**2)/2)
    else:
        expectedwork[proc] = sol[0]*dx - sol[1]*0.5*((x_max-z)**2 + (z-x_min)**2)

print(np.sum(total_work))
print(np.sum(expectedwork))
        
const = total_work[special] - expectedwork[special]
expectedwork[special] = total_work[special]

print("h = {}, m = {}, c = {}".format(sol[0], sol[1], const))
xs = decompose_domain(world_size, sol[0], sol[1], const)

plt.figure()
plt.plot((np.arange(world_size)+0.5)/world_size, total_work, '+-')
plt.plot((np.arange(world_size)+0.5)/world_size, expectedwork, 'r')
plt.plot(xs, 3*np.ones_like(xs), 'go')
plt.plot(1/np.sqrt(2), 3, 'ro')
plt.ylim([0, plt.ylim()[1]])
plt.show()

print(np.sum(total_work))
print(np.sum(expectedwork))
print(sol[0]-0.5*sol[1]*(1/2 + (1-1/np.sqrt(2))**2)+const)

if True:
    x = df_w['x']
    w = df_w['weight']
    w /= np.amax(w)
    
    plt.figure()
    plt.plot(x, w)
    plt.xlabel('x')
    plt.ylabel('Visit count')
    plt.show()