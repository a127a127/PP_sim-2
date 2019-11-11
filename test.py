import numpy as np
import tracemalloc as tm
import time 

N = 1000

start = time.time()
A_list = []
for i in range(N):
    A_list.append([])
    for j in range(N):
        A_list[i].append(j)
end = time.time()
#print(A_list)
print("Time:", end - start)
print()

#A_numpy = np.array(A_list)
start = time.time()
A_numpy = np.zeros([N, N])
for i in range(A_numpy.shape[0]):
    for j in range(A_numpy.shape[1]):
        A_numpy[i][j] = j
end = time.time()
#print(A_numpy)
print("Time:", end - start)
print()

start = time.time()
for i in range(len(A_list)):
    for j in range(len(A_list[i])):
        A_list[i][j] = j
end = time.time()
#print(A_list)
print("Time:", end - start)
print()


N = 100000
tm.start()
A = [0 for i in range(N)]
snap = tm.take_snapshot()
stats = snap.statistics('lineno')
for stat in stats:
    print(stat)
tm.stop()

# tm.start()
# start = time.time()
# AA = np.zeros(N)
# snap = tm.take_snapshot()
# stats = snap.statistics('lineno')
# for stat in stats:
#     print(stat)
# print(time.time() - start)
# tm.stop()
