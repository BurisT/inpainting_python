import time
import matplotlib.pyplot as plt
import math
import numpy as np
tic = time.time()
a=1
b=2
c= 30*9.9 + a + b
print(c)

A = np.ones((3,3,3))
B = np.ones((3,3,3))*2
C = np.ones((3,3,3))*3
D = A * 0.5

print(C*B)

p = np.clip(D,0,1)
print('\n')
print(p)

import timeit

print(timeit.timeit(stmt='FBF',setup='from main import FBF', number=1))