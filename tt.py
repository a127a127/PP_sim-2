import numpy as np
import gc
import time as tt

class A():
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.li = [1,2,3]

class B():
    def __init__(self, a):
        self.a = a
        self.lii = a.li
    def ff(self):
        del self.lii
        #del self.a.li

A = [A() for i in range(1000000)]
A = np.array(A)
start = tt.time()
for aa in A:
    aa.a = 10

print("ok", tt.time() - start)
a = input()
