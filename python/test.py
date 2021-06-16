import numpy as np

sigma_x = np.array([[0, 1],
                    [1, 0]])

unit = np.eye(2)

H_q = None

for i in range(1, 0):
    print(i)

N = 3
for i in range(1, N + 1):
    sigma_x_i = sigma_x.copy()
    for k in range(1, i):
        sigma_x_i = np.kron(unit, sigma_x_i)

    for k in range(i + 1, N + 1):
        sigma_x_i = np.kron(sigma_x_i, unit)

    if H_q is None:
        H_q = sigma_x_i
    else:
        H_q += sigma_x_i

print(H_q)

s = np.sum(sigma_x)
print(s)

i = 2
N = 2

array = np.array([i ^ (1 << n) for n in range(N)])
for n in range(N):
    print(format(i, "02b") + " ^ " + format(1 << n, "02b"))

print()
for i in range(2 ** N):
    print(format(i, "02b"))
