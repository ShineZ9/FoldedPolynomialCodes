'''
Finding the worst case condition number for FPC with matrix partitioning  parameter m=1,
where in the computing system， there are n worker nodes and s stragglers.
A is evenly divided into $p$ column blocks and A^\top is evenly divided into $p$ row blocks.
'''



import numpy as np
import itertools as it
import random
import math

p = 4;                      # The number of column blocks
k = p;                      # The recovery threshold of FPC
s = 1;                      # The number of straggler nodes
n = k + s;                  # The number of worker nodes

node_points = np.zeros(n,dtype=float) ;
'''X=[];
for i in range(0, n):
    X.append(random.uniform(-1,1));
    ss=list(set(X));
for i in range(0,n):
    node_points[i]=ss[i];'''
for i in range(0, n):
    a = (2 * i + 1) / (2 * n) * math.pi;
    node_points[i] = math.cos(a);
workers = np.array(list(range(n)));
Choice_of_workers = list(it.combinations(workers, k));
size_total = np.shape(Choice_of_workers);
total_no_choices = size_total[0];
cond_no = np.zeros(total_no_choices, dtype=float);
T = np.zeros((n, k), dtype=float);
for j in range(0, n):
    T[j, :] = (node_points[j]) ** (k - 1 + np.array(list(range(k)))) + (node_points[j]) ** (
            k - 1 - np.array(list(range(k))))
for i in range(0, total_no_choices):
    dd = list(Choice_of_workers[i]);
    cond_no[i] = np.linalg.cond(T[dd, :]);
worst_condition_number = np.max(cond_no);
pos = np.argmax(cond_no);
worst_choice_of_workers = list(Choice_of_workers[pos]);
print('Worst condition Number is %s' % worst_condition_number)
print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)