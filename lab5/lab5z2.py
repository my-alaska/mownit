import numpy as np

def newton(X,f,df,mode="ab",e=0.0000001):
    i = 0
    while True:
        Xp = np.copy(X)
        X = X - f(X) @ np.linalg.inv(df(X)).T
        i+= 1
        if (mode == "ab" and np.linalg.norm(X-Xp) < e) or (mode == "f" and np.linalg.norm(f(X)) < e):
            break
    return X, i


def f(v):
    x1, x2, x3 = v[0], v[1], v[2]
    return np.array([x1**2 + x2**2 + x3 - 1,
            2*x1**2 - x2**2 - 4*x3**2 +3,
            x1**2 + x2 + x3 - 1])

def df(v):
    x1, x2, x3 = v[0], v[1], v[2]
    return np.array([[2*x1,2*x2,1],
            [4*x1,-2*x2,-8*x3],
            [2*x1,1,1]])


print("vector distance exit:")
start = np.array([-3,3,-3],dtype = 'float64')
print(newton(start,f,df))
start = np.array([3,3,-3],dtype = 'float64')
print(newton(start,f,df))
start = np.array([-3,0,-5],dtype = 'float64')
print(newton(start,f,df))
start = np.array([3,0,-5],dtype = 'float64')
print(newton(start,f,df))
start = np.array([-5,0,3],dtype = 'float64')
print(newton(start,f,df))
start = np.array([5,0,3],dtype = 'float64')
print(newton(start,f,df))
start = np.array([5,-1,3],dtype = 'float64')
print(newton(start,f,df))

print("\nfunction value exit:")
start = np.array([-3,3,-3],dtype = 'float64')
print(newton(start,f,df,"f"))
start = np.array([3,3,-3],dtype = 'float64')
print(newton(start,f,df,"f"))
start = np.array([-3,0,-5],dtype = 'float64')
print(newton(start,f,df,"f"))
start = np.array([3,0,-5],dtype = 'float64')
print(newton(start,f,df,"f"))
start = np.array([-5,0,3],dtype = 'float64')
print(newton(start,f,df,"f"))
start = np.array([5,0,3],dtype = 'float64')
print(newton(start,f,df,"f"))

