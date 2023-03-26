'''
The Numerical Experiments of MatDot Codes over MAGIC CUBE-III cluster.
Finding  Worker Computation Time and Decoding Time.
We have n worker nodes and s=n-(2*p-1) stragglers.
A is a $t \times r$ matrix over the real number field, which is evenly divided into $p$ column blocks.
'''

from __future__ import division
from mpi4py import MPI
import sys
import warnings
import numpy as np
import itertools as it
import math
import time
import random

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    p = 8;
    k = 2*p-1;
    s = 1;
    n = k + s;
    r = 15000;                       # number of columns in matrix A
    t = 12000;                       # number of rows in matrix A
    mu = 0;
    sigma = 1;
    A = np.random.normal(mu, sigma, [t, r]);
    B = np.transpose(A);
    E = np.matmul(A, B);
    node_points = np.zeros(n, dtype=float);
    for i in range(0, n):
        a = (2 * i + 1) / (2 * n) * math.pi;
        node_points[i] = math.cos(a);
    Encoding_matrix = np.zeros((n, p), dtype=float);
    for j in range(0, n):
        Encoding_matrix[j, :] = (node_points[j]) ** np.array(list(range(p)));

    Coding_A = Encoding_matrix;
    c = int(r / p);
    W1a = {};
    for i in range(0, p):
        W1a[i] = A[:, i * c:(i + 1) * c];
    W2a = {};
    (uu, vv) = np.shape(W1a[0]);
    for i in range(0, n):
        W2a[i] = np.zeros((uu, vv), dtype=float);
        for j in range(0, p):
            W2a[i] = W2a[i] + Coding_A[i, j] * W1a[j];
    Coding_B = Encoding_matrix;
    W1b = {};
    for i in range(0, p):
        W1b[i] = np.transpose(W1a[i])
    W2b = {};
    (uu, vv) = np.shape(W1b[0]);
    for i in range(0, n):
        W2b[i] = np.zeros((uu, vv), dtype=float);
        for j in range(0, p):
            W2b[i] = W2b[i] + Coding_B[i, p - 1 - j] * W1b[j];
    work_product = {};
    for i in range(0, n):
        Ai = W2a[i];
        Bi = W2b[i];
        comm.send(Ai, dest=i + 1)
        comm.send(Bi, dest=i + 1)
    computation_time = np.zeros(n, dtype=float);
    for i in range(0, n):
        computation_time[i] = comm.recv(source=i + 1);
        work_product[i] = comm.recv(source=i + 1);
    worktime = np.average(computation_time);                               # Worker Computation Time
    print('\n')
    for i in range(0, n):
        print("Computation time for processor %s is %s" % (i, computation_time[i]))
    print('\n')
    print("Work time %s is %s" % worktime)
    computation_time.sort();
    overalltime = computation_time[k - 1];  # Overall Computation Time
    print("Overall  time %s is %s" % overalltime)
    worker_product = {};
    workers = np.array(list(range(n)));
    Choice_of_workers = list(it.combinations(workers, k));
    size_total = np.shape(Choice_of_workers);
    total_no_choices = size_total[0];
    cond_no = np.zeros(total_no_choices, dtype=float);
    T = np.zeros((n, k), dtype=float);
    for j in range(0, n):
        T[j, :] = (node_points[j]) ** (np.array(list(range(k))))
    for i in range(0, total_no_choices):
        dd = list(Choice_of_workers[i]);
        cond_no[i] = np.linalg.cond(T[dd, :]);
    worst_condition_number = np.max(cond_no);
    pos = np.argmax(cond_no);
    worst_choice_of_workers = list(Choice_of_workers[pos]);
    start_time = time.time();
    decoding_mat = np.linalg.inv(T[worst_choice_of_workers,:]);
    (g, h) = np.shape(worker_product[0])
    X = worker_product[0];
    X = X + np.transpose(X);
    CC = X.ravel();
    for i in range(1, k):
        x = worker_product[i];
        y = x.ravel();
        CC = np.concatenate((CC, y), axis=0);
    BB = np.reshape(CC, (k, -1));
    decoded_block = np.matmul(decoding_mat[p-1, :], BB);
    C = np.reshape(decoded_block, (g, -1));
    end_time = time.time();
    Decodingtime = end_time - start_time;  # Decoding Time
    print('Decoding time is %s seconds' % (Decodingtime))
    err_ls = 100 * np.square(np.linalg.norm(C - E, 'fro') / np.linalg.norm(E, 'fro'));
    print("----------------------")
    print('Error Percentage is %s ' % err_ls)

elif rank < 2:  # s stragglers have one-fifth of the speed of the non-straggling nodes.
    Ai = comm.recv(source=0)
    Bi = comm.recv(source=0)
    start_time = time.time()
    Wab = np.matmul(Ai, Bi);
    Wab = 2 * np.matmul(Ai, Bi);
    Wab = 3 * np.matmul(Ai, Bi);
    Wab = 4 * np.matmul(Ai, Bi);
    Wab = np.matmul(Ai, Bi);
    end_time = time.time();
    comp_time = end_time - start_time;
    comm.send(comp_time, dest=0)
    comm.send(Wab, dest=0)
else:
    Ai = comm.recv(source=0)
    Bi = comm.recv(source=0)
    start_time = time.time()
    Wab = np.matmul(Ai, Bi);
    end_time = time.time();
    comp_time = end_time - start_time;
    comm.send(comp_time, dest=0)
    comm.send(Wab, dest=0)


