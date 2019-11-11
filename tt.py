import numpy as np
import gc

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

A1 = A()
B1 = B(A1)
B1.lii[0] = 22
print(B1.lii)
print(A1.li)
B1.ff()
print(A1.li)
