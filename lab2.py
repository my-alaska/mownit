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


def get_polynomial_L(X, Y):
    n = len(X)
    m = [1] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i] *= X[i] - X[j]

    def f(x):
        d = [1] * n
        for i in range(n):
            for j in range(n):
                if X[i] != X[j]:
                    d[i] *= x - X[j]
        P = 0
        for i in range(n):
            P += Y[i] * (d[i] / m[i])
        return P

    return f


def get_polynomial_N(X, Y):
    n = len(X)

    def diff(X, Y):
        quotients = [[None] * n for i in range(n)]

        def curr(i, j):
            if quotients[i][j] is not None:
                pass
            elif i == j:
                quotients[i][j] = Y[i]
            else:
                f2 = curr(i + 1, j)
                f1 = curr(i, j - 1)
                quotients[i][j] = (f2 - f1) / (X[j] - X[i])
            return quotients[i][j]

        curr(0, n - 1)
        return quotients

    Q = diff(X, Y)

    def f(x):
        result = 0
        for i in range(n - 1, 0, -1):
            result += Q[0][i]
            result *= (x - X[i - 1])
        result += Q[0][0]
        return result

    return f


def function_to_points(X, f): return [f(X[i]) for i in range(len(X))]


def plot(a,b,n,f,cheb,L):
    if cheb:
        x = np.array(chebyshev(n, a, b))
    else:
        x = np.array(even(n, a, b))

    y = f(x)
    if L:
        f2 = get_polynomial_L(x,y)
    else:
        f2 = get_polynomial_N(x,y)

    x2 = np.linspace(a,b,2000)
    y2 = f2(x2)
    y3 = f(x2)
    plt.plot(x2,y2,'g')
    plt.plot(x2, y3, 'b')
    plt.scatter(x,y,c='r',s=15,marker = 'o')
    plt.show()


def f_compare_max(X1,X2):
    diff = np.abs(X1 - X2)
    return np.max(diff)


def f_compare_sqr(X1,X2):
    diff = np.square(X1 - X2)
    return np.sum(diff)


# (a,b) - scope, f - function, n - number of nodes, N - comparison precision
def evaluate(a,b,f,n,N=1000):
    # Chebyshev's zeros
    xc = np.array(chebyshev(n, a, b))
    yc = f(xc)

    # even distribution
    xe = np.array(even(n, a, b))
    ye = f(xe)

    fcl = get_polynomial_L(xc, yc)
    fcn = get_polynomial_N(xc, yc)
    fel = get_polynomial_L(xe, ye)
    fen = get_polynomial_N(xe, ye)

    X = np.linspace(a, b, N)
    Ycl = fcl(X)
    Ycn = fcn(X)
    Yel = fel(X)
    Yen = fen(X)



def f(x): return np.exp(-3*np.sin(3*x))

plot(-2*pi,pi,30,f,True,False)