from math import cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt

def secant(a,b,f,mode = "f",e=0.00001):
    i = 0
    while True:
        i+=1

        next = a - f(a) * (b-a)/(f(b)-f(a))

        if mode == "f" and abs(f(next)) < e:
            return next, i
        elif mode == "ab" and abs(b-a) <= e:
            return next, i

        if f(next) * f(b) > 0 and abs(b-next) >= e:
            b = next
        else:
            a = next

    # return a,i

def newton(a,f,df, mode = "ab",e=0.00001):
    i=0
    b = a-0.1
    while True:
        a,b = b, b-f(b)/df(b)
        if mode == "f" and abs(f(a)) < e:
            return a, i
        elif mode == "ab" and abs(b-a) < e:
            return a, i
        i+=1
    # return b,i

def f(x):
    return 30*x*np.exp(-10) - 30*np.exp(-10*x) + 1/30
def df(x):
    return 30*np.exp(-10)+300*np.exp(-10*x)
a,b = 0.2,1.2

x = np.linspace(a, b, 2000)
y = f(x)
plt.plot(x, y)
plt.plot(x, np.zeros(len(x)))
plt.show()


print("\nnewton at 0.2 points")
print(newton(a,f,df,"ab",0.0001))
print(newton(a,f,df,"ab",0.00001))
print(newton(a,f,df,"ab",0.000001))
print(newton(a,f,df,"ab",0.0000001))

print("\nnewton at 0.2 function")
print(newton(a,f,df,"f",0.0001))
print(newton(a,f,df,"f",0.00001))
print(newton(a,f,df,"f",0.000001))
print(newton(a,f,df,"f",0.0000001))


print("\nnewton at 1.2 points")
print(newton(b,f,df,"ab",0.0001))
print(newton(b,f,df,"ab",0.00001))
print(newton(b,f,df,"ab",0.000001))
print(newton(b,f,df,"ab",0.0000001))

print("\nnewton at 1.2 function")
print(newton(b,f,df,"f",0.0001))
print(newton(b,f,df,"f",0.00001))
print(newton(b,f,df,"f",0.000001))
print(newton(b,f,df,"f",0.0000001))


print("secants points")
print(secant(a,b,f,"ab",0.0001))
print(secant(a,b,f,"ab",0.00001))
print(secant(a,b,f,"ab",0.000001))
print(secant(a,b,f,"ab",0.0000001))

print("secants function")
print(secant(a,b,f,"f",0.0001))
print(secant(a,b,f,"f",0.00001))
print(secant(a,b,f,"f",0.000001))
print(secant(a,b,f,"f",0.0000001))

print("\ndifferent points for secant methode, points and function")
for i in range(5):
    print(round(a + i * 0.1, 1), b, ":", secant(a + i * 0.1, b, f, "ab", 0.0000001),secant(a + i * 0.1, b, f, "f", 0.0000001))
for i in range(1,6):
    print(a, round(b - i * 0.1, 1), ":", secant(a, b - i * 0.1, f, "ab", 0.0000001),secant(a, b - i * 0.1, f, "f", 0.0000001))

print("\ndifferent points for newton's methode, points and function")
for i in range(0,11):
    print(round(a + i * 0.1, 1), ":", newton(a + i * 0.1, f, df, "ab", 0.0000001), newton(a + i * 0.1, f, df, "f", 0.0000001))




