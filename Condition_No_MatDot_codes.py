'''
Finding the worst case condition number for MatDot codes,
where in the computing system， there are n worker nodes and s stragglers.
A is evenly divided into $p$ column blocks and A^\top is evenly divided into $p$ row blocks.
'''



import numpy as np
import itertools as it
import random
import math

p = 4 ;                     # The number of column blocks
k = 2*p-1;                  # The recovery threshold of MatDot codes
s = 1;                      # The number of straggler nodes
n = k + s;                  # The number of worker nodes

node_points = np.zeros(n,dtype=float) ;
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
    T[j, :] =(node_points[j]) ** (np.array(list(range(k))))
for i in range(0, total_no_choices):
    dd = list(Choice_of_workers[i]);
    cond_no[i] = np.linalg.cond(T[dd,:]);
worst_condition_number = np.max(cond_no);
pos = np.argmax(cond_no);
worst_choice_of_workers = list(Choice_of_workers[pos]);
print('Worst condition Number is %s' % worst_condition_number)
print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)