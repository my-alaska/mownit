# Lab 1

The goal of the first laboratory exercise was to present the loss of numerical precision. As an example the two following functions were used

 $f(x) = \sqrt{x^2 + 1}-1$ 
 $g(x) = \dfrac{x^2}{\sqrt{x^2+1}+1}$

Those functions are mathematically equal but the program shows clearly that the way in which we compute them on a numerical machine can heavily affect the precision

The code used in the laboratory can be found in `lab1.py` file