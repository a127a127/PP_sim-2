import numpy as np
import gc
import time as tt
import collections

class A():
    def __init__(self):
        self.a = 1
        self.b = [1,2,3]
        self.c = True

class B():
    def __init__(self, a):
        self.a = a
        self.lii = a.li
    def ff(self):
        del self.lii
        #del self.a.li




start = tt.time()
d = collections.deque()
d.append(1)
d.append(1)
d.append(1)
d.append(1)
d.append(1)
for i in d:
    print(i)

print("ok", tt.time() - start)
a = input()
