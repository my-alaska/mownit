import numpy as np
import matplotlib.pyplot as plt
import time


# function generating the matrix of size n from the task
def get_matrix(n, k=11, m=3):
    result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                result[i, j] = k
            else:
                result[i, j] = 1 / (np.abs(i-j) + m)
    return result


# vector of numbers that are either -1 or 1
def generate_ones(n):
    vector = [1] * n
    for i in range(n):
        if ((i * 6) % 7) % 2 == 0:
            vector[i] *= -1
    return np.array(vector)


# function for solving matrix from the task for given exit condition, matrix size, starting vector and epsilon
def solve(n, x_start, stop="d", e=0.0001):
    A = get_matrix(n)
    x_result = generate_ones(n)
    b = A @ x_result

    g = b / A.diagonal()
    H = -1 * A / A.diagonal().reshape((-1, 1)) + np.identity(len(A))

    if np.max(np.abs(np.linalg.eigvals(H))) >= 1:
        return None

    t0 = time.perf_counter()
    iter = 0
    x = x_start
    while True:
        prevx = x
        x = g + H @ x
        if stop == "d" and np.linalg.norm(prevx - x) < e:
            break
        elif stop == "f" and np.linalg.norm(A @ x - b) < e:
            break
        iter += 1
    t = time.perf_counter() - t0
    return x, np.linalg.norm(x - x_result), t, iter+1

# Function for calculating spectral radius
def get_radius(n):
    A = get_matrix(n)
    H = -1 * A / A.diagonal().reshape((-1, 1)) + np.identity(len(A))
    return np.max(np.abs(np.linalg.eigvals(H)))


def clear(i):
    return [None] * 20,[None] * 20,[None] * 20,[None] * 20
sizes = [(i + 1) * 25 for i in range(20)]


# testing function. Start function is a python function reference for generating starting vector
# exit is either either "d" or "f" string depending on the exit condition
# e is our epsilon that approximates zero
def test(start_function, exit, e):
    x,norm,t,iter = clear(20)
    for i in range(20):
        n = sizes[i]
        x_start = start_function(n)
        result = solve(n, x_start, exit, e)
        if result is not None:
            x[i], norm[i], t[i], iter[i] = result

    print("results for", exit, e)
    for i in range(20):
        print(norm[i])
    print()
    for i in range(20):
        print(round(t[i],5))
    print()
    for i in range(20):
        print(iter[i])
    print("\n\n\n")



def f(n):
    return 2*generate_ones(n)
print("2*ones")
for i in range(5):
    test(f, "d", 0.0001 * 10 ** (-i))
    test(f, "f", 0.0001 * 10 ** (-i))

def f(n):
    return -2000000*generate_ones(n)
print("-2000000*ones")
for i in range(5):
    test(f, "d", 0.0001 * 10 ** (-i))
    test(f, "f", 0.0001 * 10 ** (-i))


for n in sizes:
    print(get_radius(n))