from scipy.integrate import complex_ode
from scipy.linalg import eigh
import numpy as np
N = 5
E = np.random.normal(0, scale=N / 2, size=2**N)

Tau = 1


def scheduleE(time):
    return time / Tau


def scheduleG(time):
    return (Tau - time) / Tau


def create_tfim(time=0, hamiltonian=None):
    # Set diagonal part
    v = scheduleE(time)
    if hamiltonian is None:
        hamiltonian = np.diag(v * E)
    else:
        for i in range(2**N):
            hamiltonian[i, i] = v * E[i]

    # Set off-diagonal part
    g = - 1 * scheduleG(time)
    for i in range(2**N):
        for n in range(N):
            j = i ^ (1 << n)
            hamiltonian[i, j] = g
    return hamiltonian


H = None
step = 0.01
time_steps = [step * i for i in range(int(Tau / step) + 1)]
for i, t in enumerate(time_steps):
    H = create_tfim(t, H)
    evals_all, evecs_all = eigh(H)
    # ADDITIONAL CODE HERE


def amp2prob(vec):
    p = [np.abs(z)**2 for z in vec]
    return np.array(p)


def genrate_diffeq(idx, offdiag_indices, time, vec):
    v = E[idx] * scheduleE(time)
    g = -1 * scheduleG(time)
    return v * vec[idx] + g * np.sum(vec[offdiag_indices])


diffeq_array = [lambda vec, t, _i=i, _ij=np.array([i ^ (1 << n) for n in range(N)]):
                genrate_diffeq(_i, _ij, t, vec) for i in range(2**N)]


def simdiffeq_rhs(t, vec):
    return np.array([- 1j * f(vec, t) for f in diffeq_array],
                    dtype=np.complex_)


vec0 = np.array([2**(-N / 2)] * 2**N, dtype=np.complex_)
r = complex_ode(simdiffeq_rhs)
r.set_initial_value(vec0)

steps = 100
i = 0
t = 0
while r.successful() and i <= steps:
    v = r.integrate(t) if t != 0 else vec0
    # ADDITIONAL CODE HERE
    t += Tau / steps
