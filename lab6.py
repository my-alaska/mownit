import numpy as np
import time

# generates a badly conditioned matrix
def get_hilbert(n):
    matrix = np.ones((n, n))
    vector = np.arange(2,n+2,dtype = 'float64')
    ones_vector = np.ones(n)
    for i in range(1,n):
        matrix[i] /= vector
        vector += ones_vector
    return matrix

# generates better matrix
def get_better(n):
    result = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j >= i:
                result[i][j] = 2*(i+1)/(j+1)
            else: result[i][j] = result[j][i]
    return np.array(result)


#generates my own matrix tasked by the teacher
def get_tasked(n,k=8,m=4,dtype = "float64"):
    result = np.diag(np.array([k]*n,dtype = dtype))
    for i in range(n-1):
        result[i, i+1] = 1/(i+1+m)
        result[i+1, i] = k/(i+3+m)
    return result


def generate_ones(n):
    vector = [1]*n
    for i in range(n):
        if ((i*6)%7) %2 == 0:
            vector[i] *= -1
    return np.array(vector)

def solve(n, matrix_function):
    A,x = matrix_function(n), generate_ones(n)
    b = A@x
    print("condition number: ", np.linalg.cond(A))
    # print(np.linalg.cond(A))
    t0 = time.perf_counter()
    for i in range(n):
        for j in range(i+1,n):
            b[j] -= b[i]*A[j,i]/A[i,i]
            A[j] -= A[i]*A[j,i]/A[i,i]
            A[j][i] = 0
    result = np.zeros(n)
    result[n-1] = b[n-1]/A[n-1][n-1]
    for i in range(n-2,-1,-1):
        result[i] = (b[i]-np.sum(result*A[i]))/A[i][i]
    t = time.perf_counter() - t0
    print(f"time: {round(t, 6)} s")
    # print(f"{round(t, 6)}")
    differences = x - result
    print("max difference:  ", np.max(differences))
    print("norm difference: ", np.linalg.norm(differences))
    # print(np.linalg.norm(differences))
    return result

def solve_thomas(n, matrix_function):
    A, x = matrix_function(n), generate_ones(n)
    d = A @ x
    print("condition number: ", np.linalg.cond(A))
    # print(np.linalg.cond(A))
    t0 = time.perf_counter()
    cb = np.zeros(n)
    db = np.zeros(n)
    cb[0],db[0] = A[0,1]/A[0,0], d[0]/A[0,0]
    for i in range(1, n):
        db[i] = (d[i] - A[i, i - 1] * db[i - 1]) / (A[i, i] - A[i, i - 1] * cb[i - 1])
        if i == n-1: break
        cb[i] = A[i, i + 1] / (A[i, i] - A[i, i - 1] * cb[i - 1])
    result = np.zeros(n)
    result[n-1] = db[n-1]
    for i in range(n-2,-1,-1):
        result[i] = db[i] - cb[i]*result[i+1]
    t = time.perf_counter() - t0
    print(f"time: {round(t,6)} s")
    # print(f"{round(t, 6)}")
    differences = x - result
    print("max difference:  ", np.max(differences))
    print("norm difference: ", np.linalg.norm(differences))
    # print(np.linalg.norm(differences))
    return result

for i in range(2,21,3):
    print(f"solving task 1 for {i}")
    solve(i,get_hilbert)
    print()

for i in range(2,21,3):
    print(f"solving task 2 for {i}")
    solve(i,get_better)
    print()



for i in range(2,21,3):
    print(f"\nsolving with default methode for {i}")
    solve(i, get_tasked)
    print(f"\nsolving with Thomas' methode for {i}")
    solve_thomas(i, get_tasked)
