from math import cos, pi
import matplotlib.pyplot as plt
import numpy as np


def even(n, a, b):
    n -= 1
    d = (b - a) / n
    result = [0] * (n + 1)
    result[0] = a
    result[n] = b
    for i in range(1, n):
        result[i] = result[i - 1] + d
    return result


def lsquares_tr(X, Y, m, a, b, w=None):
    def to2pi(x):
        return (2*pi*x - 2*pi*a)/(b-a)

    X = to2pi(X)

    n = len(X)
    if w is None:
        w = np.array([1] * n)
    matrix = np.zeros((m * 2 + 1, m * 2 + 1))
    r_vector = np.empty(2 * m + 1)

    matrix[0, 0] = w.sum()
    for i in range(m):
        matrix[0, 2 * i + 1] = (w * np.sin((i + 1) * X)).sum()
        matrix[0, 2 * i + 2] = (w * np.cos((i + 1) * X)).sum()
    r_vector[0] = (Y * w).sum()
    for i in range(m):
        matrix[2 * i + 1, 0] = (w * np.sin((i + 1) * X)).sum()
        matrix[2 * i + 2, 0] = (w * np.cos((i + 1) * X)).sum()
        for j in range(m):
            matrix[2 * i + 1, 2 * j + 1] = (w * np.sin((i + 1) * X) * np.sin((j + 1) * X)).sum()
            matrix[2 * i + 1, 2 * j + 2] = (w * np.sin((i + 1) * X) * np.cos((j + 1) * X)).sum()

            matrix[2 * i + 2, 2 * j + 1] = (w * np.cos((i + 1) * X) * np.sin((j + 1) * X)).sum()
            matrix[2 * i + 2, 2 * j + 2] = (w * np.cos((i + 1) * X) * np.cos((j + 1) * X)).sum()
        r_vector[2 * i + 1] = (w * np.sin((i + 1) * X) * Y).sum()
        r_vector[2 * i + 2] = (w * np.cos((i + 1) * X) * Y).sum()

    c = np.linalg.solve(matrix, r_vector)

    def f(x):
        x = to2pi(x)
        result = np.array([None] * (2 * m + 1))
        result[0] = 1
        for i in range(m):
            result[2 * i + 1] = (np.sin((i + 1) * x))
            result[2 * i + 2] = (np.cos((i + 1) * x))
        result *= c
        return np.sum(result)

    return f


def function_to_points(X, f): return [f(X[i]) for i in range(len(X))]


def f_compare_max(X1, X2):
    diff = np.abs(X1 - X2)
    return np.max(diff)


def f_compare_sqr(X1, X2):
    diff = np.square(X1 - X2)
    return np.sum(diff)


def f(x): return np.exp(-3 * np.sin(3 * x))


X = np.array(even(51, -2 * pi, pi)) # węzły
Y = f(X)

p = lsquares_tr(X, Y, 20,-2 * pi,pi) # stopień
x = np.linspace(-2 * pi, pi, 2000)
y1 = f(x)
y2 = p(x)

plt.plot(x, y1, 'g')
plt.plot(x, y2, 'b')
plt.scatter(X, Y, c='r', s=15, marker='o')
plt.show()
