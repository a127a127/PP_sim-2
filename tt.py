import numpy as np
import gc

class A():
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

#aa = [A() for i in range(1000000)]
aa = [[1,2,3] for i in range(1000000)]
aa = np.array(aa)
print("dsafsd")
b = input()


