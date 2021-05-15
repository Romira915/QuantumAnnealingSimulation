import numpy as np

N = 5

for i in range(2**N):
    # array = np.array([i ^ (1 << n) for n in range(N)])
    for n in range(N):
        print(i ^ (1 << n), end="")
