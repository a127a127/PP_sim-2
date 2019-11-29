import numpy as np
import gc
import time as tt
import collections
import pandas as pd
import copy

A = 5
B = 4
C = 3

print("A:", A, "B:", B, "C:", C)
print("(A - B + 1) // C =", (A - B + 1) // C)
print("(A - B) // C + 1 =", (A - B) // C + 1)

a = 1
b = 0
for i in range(2):
    print("i", i)
    if a == 0:
        if b == 0:
            print("1")
        else:
            print("2")
    elif a == 1:
        print("3")
    