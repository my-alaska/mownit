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

def lsquares(X,Y,m,w=None):
    n = len(X)
    if w is None:
        w = np.array([1]*n)
    matrix = np.zeros((m+1, m+1))
    r_vector = np.empty(m+1)
    tempsum = w.copy()

    for i in range(m+1):
        matrix[0,i] = tempsum.sum()
        r_vector[i] = (tempsum*Y).sum()
        tempsum = X*tempsum
    for i in range(1,m+1):
        matrix[i,m] = tempsum.sum()
        tempsum = X*tempsum
    for i in range(1,m+1):
        matrix[i][0:m] = matrix[i-1][1:m+1]
    a = np.linalg.solve(matrix, r_vector)

    def f(x):
        xarray = np.vstack([x]*(m+1)).T
        powers = np.arange(0,m+1)
        xarray = np.power(xarray,powers)*a
        return np.sum(xarray,1)

    return f





def function_to_points(X, f): return [f(X[i]) for i in range(len(X))]


def f_compare_max(X1,X2):
    diff = np.abs(X1 - X2)
    return np.max(diff)


def f_compare_sqr(X1,X2):
    diff = np.square(X1 - X2)
    return np.sum(diff)


def f(x): return np.exp(-3*np.sin(3*x))


X = np.array(even(200,-2*pi,pi))
Y = f(X)

p = lsquares(X,Y,25)
x = np.linspace(-2*pi,pi,2000)
y1 = f(x)
y2 = p(x)

plt.plot(x, y1, 'g')
plt.plot(x, y2, 'b')
plt.scatter(X, Y, c='r', s=15, marker='o')
plt.show()