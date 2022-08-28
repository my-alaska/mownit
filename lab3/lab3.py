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


def spline3(X, Y, zero = False):
    n = len(X)
    h = np.array([(X[i+1]-X[i]) for i in range(n-1)])

    matrix = np.zeros((n,n))
    for i in range(1,n-1):
        matrix[i][i-1] = h[i-1]
        matrix[i][i]= 2*(h[i-1]+h[i])
        matrix[i][i+1] = h[i]
    if zero:
        matrix[0][0] = 1
        matrix[n-1][n-1] = 1
    else:
        matrix[0][0],matrix[0][1] = -h[0], h[0]
        matrix[n-1][n-2],matrix[n-1][n-1] = h[n-2], -h[n-2]

    def rec(i,j):
        if i == j:
            return Y[i]
        else:
            return (rec(i+1, j)-rec(i, j-1))/(X[j]-X[i])


    delta = [(Y[i + 1] - Y[i]) / h[i] for i in range(n - 1)]
    deltaVector = np.zeros(n)
    for i in range(1,n-1):
        deltaVector[i] = delta[i]-delta[i-1]
    if zero:
        deltaVector[0] =0
        deltaVector[n - 1] = 0
    else:
        deltaVector[0] = (h[0] ** 2) * rec(0, 3)
        deltaVector[n-1]= -1*(h[n-2]**2)*rec(n-4,n-1)

    s = np.linalg.solve(matrix,deltaVector)



    def f(x):
        i = 1
        while x > X[i]:
            i+=1
        i -= 1
        b = (Y[i + 1] - Y[i]) / h[i] - h[i] * (s[i + 1] + 2 * s[i])
        c = 3 * s[i]
        d = (s[i + 1] - s[i]) / h[i]
        y = Y[i] + b*(x-X[i]) + c*(x-X[i])**2 + d*(x-X[i])**3
        return y
    return f




def spline2(X, Y, dxstart = None):
    n = len(X)
    h = np.array([(X[i+1]-X[i]) for i in range(n-1)])

    matrix = np.zeros(((n-1)*3,(n-1)*3))
    for i in range(1,3*(n-1)):
        k = i%3
        match k:
            case 1:
                matrix[i][i-1] = 1
            case 2:
                matrix[i][i-2] = 1
                matrix[i][i-1] = h[i//3]
                matrix[i][i] = h[i//3]**2
            case 0:
                matrix[i][i - 2] = 1
                matrix[i][i - 1] = 2*h[(i-1) // 3]
                matrix[i][i + 1] = -1
    if dxstart is not None:
        matrix[0][1] = 1
    else:
        matrix[0][2] = 1

    r_vector = [None]*(3*(n-1))
    for i in range(0,(n-1)):
        r_vector[3*i] = 0
        r_vector[3*i+1] = Y[i]
        r_vector[3 * i + 2] = Y[i+1]
    if dxstart is not None:
        r_vector[0] = dxstart
    s = np.linalg.solve(matrix,r_vector)
    s = s.reshape((-1,3))

    def f(x):
        i = 1
        while x > X[i]:
            i+=1
        i -= 1
        a = s[i][0]
        b = s[i][1]
        c = s[i][2]
        y = a + b*(x-X[i]) + c*(x-X[i])**2
        return y
    return f


def function_to_points(X, f): return [f(X[i]) for i in range(len(X))]


def f_compare_max(X1,X2):
    diff = np.abs(X1 - X2)
    return np.max(diff)


def f_compare_sqr(X1,X2):
    diff = np.square(X1 - X2)
    return np.sum(diff)


def f(x): return np.exp(-3*np.sin(3*x))
dx_a = -9



X = np.array(even(8,-2*pi,pi))
Y = f(X)

p = spline3(X,Y)

x = np.linspace(-2*pi,pi,2000)
y1 = f(x)
y2 = np.zeros(2000)
for i in range(2000):
    y2[i] = p(x[i])

plt.plot(x, y1, 'g')
plt.plot(x, y2, 'b')
plt.scatter(X, Y, c='r', s=15, marker='o')
plt.show()