"""Evaluate sqrt(x^2+1)-1 and x^2/(sqrt(x^2+1)+1)"""

import numpy as np
# print(np.finfo(np.double))
np.set_printoptions(precision=16)
n=20


# eights is an array of subsequent negative power of 8 (x in the task description)
exponents = -1 * np.arange(1, n+1, dtype=np.double)
powers = np.power(8, exponents, dtype=np.double)
print("\nnegative powers of 8\n", powers)


# sqrs is an array of squared values of numbers from "eights" array (x^2 in the taks description)
sqrs = powers * powers
print("\nsquares\n", sqrs)
print("\nsquares augmented by 1\n", sqrs + 1)


# roots: sqrt(x^2+1) from task description
roots = np.sqrt(sqrs + 1, dtype = np.double)
print("\nroots\n", roots)


# both resulting arrays computed in two different ways (  sqrt(x^2+1)-1  ,  x^2/(sqrt(x^2+1)+1)  )
res1 = roots - 1
res2 = sqrs / (roots + 1)
print("\nresults 1\n", res1)
print("\nresults 2\n", res2)


# difference between values from two different resulting arrays
diff = res1 - res2
print("\ndifference\n", diff)


# precision loss at computing res1
checking_double = np.double(1.0000000000000018)  - (1 + np.double(2**-49))
print("\n 1.0000000000000018== 1+2^-49",checking_double == 0)