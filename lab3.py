from math import cos, pi
import matplotlib.pyplot as plt
import numpy as np


def chebyshev(n, a, b):
    result = [0] * n
    for i in range(n):
        result[i] = (a + b) / 2 + ((b - a) / 2) * cos((2 * i + 1) * pi / (2 * n))
    return result


def even(n, a, b):
    n -= 1
    d = (b - a) / n
    result = [0] * (n + 1)
    result[0] = a
    result[n] = b
    for i in range(1, n):
        result[i] = result[i - 1] + d
    return result


def hermite(X, Y, Yd):
    n = 2*len(X)

    def diff():
        quotients = [[None] * (n) for i in range(n)]
        quotients[0][0]=Y[0]
        def curr(i, j):
            if quotients[i][j] is not None:
                pass
            elif i == j:
                quotients[i][j] = Y[i//2]
            elif i % 2 == 0 and j == i+1:
                quotients[i][j] = Yd[i//2]
            else:
                f2 = curr(i + 1, j)
                f1 = curr(i, j - 1)
                quotients[i][j] = (f2 - f1) / (X[j//2] - X[i//2])
            return quotients[i][j]

        curr(0, n - 1)
        return quotients

    Q = diff()

    def f(x):
        result = 0
        for i in range(n - 1, 0, -1):
            result += Q[0][i]
            result *= (x - X[(i - 1)//2])
        result += Q[0][0]
        return result

    return f


def function_to_points(X, f): return [f(X[i]) for i in range(len(X))]


def f_compare_max(X1,X2):
    diff = np.abs(X1 - X2)
    return np.max(diff)


def f_compare_sqr(X1,X2):
    diff = np.square(X1 - X2)
    return np.sum(diff)


def f(x): return np.exp(-3*np.sin(3*x))
def fd(x): return -9*np.cos(3*x)*np.exp(-3*np.sin(3*x))

X = np.array(chebyshev(15,-2*pi,pi))
Y = f(X)
Yd = fd(X)

p = hermite(X,Y,Yd)

x = np.linspace(-2*pi,pi,2000)
y1 = f(x)
y2 = p(x)

plt.plot(x, y1, 'g')
plt.plot(x, y2, 'b')
plt.scatter(X, Y, c='r', s=15, marker='o')
plt.show()