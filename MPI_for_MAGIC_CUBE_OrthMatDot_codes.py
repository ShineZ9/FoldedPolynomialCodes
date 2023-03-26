'''
The Numerical Experiments of OrthMatDot Codes over MAGIC CUBE-III cluster.
Finding  Worker Computation Time and Decoding Time.
We have n worker nodes and s=n-(2*p-1) stragglers.
A is a  $t \times r$ matrix over the real number field, which is evenly divided into $ka$ column blocks.
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
    ka = 8;                         # A is evenly divided into $ka$ column blocks.
    s = 1;
    k = 2 * ka - 1;
    n = k + s;
    r = 15000;                       # number of columns in matrix A
    t = 12000;                       # number of rows in matrix A;
    mu = 0;
    sigma = 1;
    A = np.random.normal(mu, sigma, [t, r]);
    B = np.transpose(A);
    E = np.matmul(A, B);

    node_points = np.zeros(n, dtype=float);
    for i in range(0, n):
        a = (2 * i + 1) / (2 * n) * math.pi;
        node_points[i] = math.cos(a);

    workers = np.array(list(range(n)));
    Choice_of_workers = list(it.combinations(workers, k));
    size_total = np.shape(Choice_of_workers);
    total_no_choices = size_total[0];
    cond_no = np.zeros(total_no_choices, dtype=float);

    T = np.zeros((k, n), dtype=float)
    for i in range(0, k):
        T[i, :] = np.cos(i * np.arccos(node_points));
    T[0, :] = T[0, :] / np.sqrt(2);

    for i in range(0, total_no_choices):
        dd = list(Choice_of_workers[i]);
        cond_no[i] = np.linalg.cond(T[:, dd]);

    worst_condition_number = np.max(cond_no);
    pos = np.argmax(cond_no);
    worst_choice_of_workers = list(Choice_of_workers[pos]);
    nodes = node_points[worst_choice_of_workers];

    Encoding_matrix = np.zeros((n, ka), dtype=float)
    for i in range(0, ka):
        Encoding_matrix[:, i] = np.cos(i * np.arccos(node_points));
    Encoding_matrix[:, 0] = Encoding_matrix[:, 0] / np.sqrt(2);

    Coding_A = Encoding_matrix;
    c = int(r / ka);
    W1a = {};
    for i in range(0, ka):
        W1a[i] = A[:, i * c:(i + 1) * c];

    W2a = {};
    (uu, vv) = np.shape(W1a[0]);
    for i in range(0, n):
        W2a[i] = np.zeros((uu, vv), dtype=float);
        for j in range(0, ka):
            W2a[i] = W2a[i] + Coding_A[i, j] * W1a[j];

    work_product = {};
    sending_time = np.zeros(n, dtype=float);
    for i in range(0, n):
        Ai = W2a[i];
        start = time.time();
        comm.send(Ai, dest=i + 1)
        end = time.time();
        sending_time[i] = end - start
    computation_time = np.zeros(n, dtype=float);

    for i in range(0, n):
        computation_time[i] = comm.recv(source=i + 1);
        work_product[i] = comm.recv(source=i + 1);
    worktime = np.average(computation_time);

    print('\n')
    print("Work time is %s" % worktime)                           # Worker Computation Time
    computation_time.sort();
    overalltime = computation_time[k - 1];                        # Overall Computation Time
    print("Overall  time is %s" % overalltime)

    worker_product = {};
    Coding_matrix2 = T[:, worst_choice_of_workers];

    start_time = time.time();
    decoding_mat = np.linalg.inv(Coding_matrix2);

    (g, h) = np.shape(worker_product[0])
    X = worker_product[0];
    CC = X.ravel();

    for i in range(1, k):
        x = worker_product[i];
        y = x.ravel();
        CC = np.concatenate((CC, y), axis=0);
    BB = np.reshape(CC, (k, -1));
    BB = np.transpose(BB);
    node_points1 = np.zeros(ka, dtype=float);
    for i in range(0, ka):
        a = (2 * i + 1) / (2 * ka) * math.pi;
        node_points1[i] = math.cos(a);
    G = np.zeros((k, ka), dtype=float)
    for i in range(0, k):
        G[i, :] = np.cos(i * np.arccos(node_points1));
    G[0, :] = G[0, :] / np.sqrt(2);
    cc = np.zeros(k, dtype=float)
    for i in range(0, ka):
        cc = cc + 2 * G[:, i] / ka;
    decoded_block = np.matmul(BB, decoding_mat);
    decoded_block1 = np.matmul(decoded_block, cc);
    C = np.reshape(decoded_block1, (g, -1));
    end_time = time.time();
    print('Decoding time is %s seconds' % (end_time - start_time))                           # Decoding Time
    err_ls = 100 * np.square(np.linalg.norm(C - E, 'fro') / np.linalg.norm(E, 'fro'));
    print('Error Percentage is %s ' % err_ls)
elif rank < 2:  # s straggler  workers have one-fifth of the speed of the non-straggling nodes.
    Ai = comm.recv(source=0)
    Bi = np.transpose(Ai);
    start_time = time.time()
    Wab = np.matmul(Ai, Bi);
    Wab = 2 * np.matmul(Ai, Bi);
    Wab = 3 * np.matmul(Ai, Bi);
    Wab = 4 * np.matmul(Ai, Bi);
    Wab = np.matmul(Ai, Bi);
    end_time = time.time();
    comp_time = end_time - start_time;
    comm.send(comp_time, dest=0)
else:
    Ai = comm.recv(source=0)
    Bi = np.transpose(Ai)
    start_time = time.time()
    Wab = np.matmul(Ai, Bi);
    end_time = time.time();
    comp_time = end_time - start_time;
    comm.send(comp_time, dest=0)
    comm.send(Wab, dest=0)



