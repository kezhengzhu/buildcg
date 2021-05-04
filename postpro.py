#!/usr/bin/env python
import numpy as np
from numpy import fft
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import pickle
import re
from numba import jit, njit
import time

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)
    return

def read_xvg(filename, pandas=False, x_name='t'):
    # Reads an xvg file and tries to append it to x and y.
    # Ensures that it works on both lists and ndarrays
    if (not isinstance(filename, str)) or filename[-4:] != ".xvg":
        raise Exception("read_xvg failed due to wrong input or wrong file extension")
    f = open(filename, 'r')
    L = f.readlines()
    f.close()

    total_lines = len(L)
    i = 0
    line = L[0]
    if pandas: columns = [x_name]
    while line.startswith(('#', '@')):
        i += 1
        line = L[i]
        if pandas and line.startswith('@ s'):
            c = re.search('(?<=\")[\s\S]+(?=\")', line)
            columns.append(c.group(0))

    L = L[i:]
    rows = len(L)
    cols = len(L[0].strip().split())

    result = np.zeros([rows,cols])
    for i in range(rows):
        line = L[i].strip().split()
        result[i,:] = [float(x) for x in line]

    print(f"{filename} loading done!")
    return result if not pandas else pd.DataFrame(result, columns=columns)

def fftacf(p, subtract_mean=True):
    '''
    Calculating autocorrelation function of the array using Fast-Fourier Transform
    '''
    # checkerr(p.ndim == 2, "Use numpy 2d arrays")
    trun = np.size(p, axis=0)
    cols = np.size(p, axis=1)
    if subtract_mean:
        m = np.mean(p, axis=0)
        p = p - m

    p2 = np.row_stack([p, np.zeros([trun, cols])])
    x = abs(fft.fft(p2, axis=0))**2
    p2 = fft.ifft(x, axis=0).real

    return p2[:trun, :]

def acfloop(p, lags):
    nlen = np.size(p, axis=0)
    ybar = np.mean(p, axis=0)
    assert(lags <= nlen)
    result = np.zeros(lags) if len(p.shape)==1 else np.zeros((lags, np.size(p,axis=1)))

    for k in range(lags):
        upper = nlen - k
        prod = (p[0:upper]-ybar) * (p[k:upper+k]-ybar)
        result[k] = np.sum(prod, axis=0) / (nlen-1)

    return result

def main():
    # print(len(read_xvg("LJ/t1/test.xvg")))
    # print(len(read_xvg("LJ/t1/test.xvg", pandas=True)))
    N = 10000001
    n = 3000001
    itn = 1000000

    start = time.time()
    pxy = read_xvg('LJ/t1/pressure_offdiag.xvg')
    print(f"read_xvg took {time.time()-start:0.3f} s")

    T = (N-n) // itn + 1
    P = np.column_stack([pxy[i*itn:i*itn+n,1:] for i in range(T)])

    start = time.time()
    acf = fftacf(P, subtract_mean=False)
    print(f"acf done! time taken {time.time()-start:0.3f} s")

    # fw = open("LJ/t1/acf.pickle", "wb")
    # pickle.dump(acf, fw)
    # fw.close()
    # print("acf.pickle saved!")

    start = time.time()
    B = cumtrapz(acf, x=pxy[:n,0], axis=0)

    rhos = 0.4
    vol = 1000 * pow(0.3,3) / rhos
    temp = 300

    eta = (vol*1e-27)*B*1e-2/(temp*1.38064852e-23)
    print(f"eta done! time taken {time.time()-start:0.3f} s")
    # fw = open("LJ/t1/eta.pickle", "wb")
    # pickle.dump(eta, fw)
    # fw.close()
    # print("eta.pickle saved!")
    # fb = open("LJ/t1/eta.pickle", "rb")
    # eta = pickle.load(fb)
    # fb.close()
    ave = np.mean(eta, axis=1)
    fig, ax = plt.subplots()
    skip = eta[0:n:n//1000, :]
    avg = ave[0:n:n//1000]
    rows = np.size(skip, axis=0)
    t = np.linspace(0,6000, rows)
    ax.plot(t, skip, '--')
    ax.plot(t, avg, 'k-', linewidth=2.5)
    plt.show()
    return

if __name__ == '__main__':
    main()
