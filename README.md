# Data-Driven-Forecasting-Taylor-Method
There are two python files in this folder which work very similarly. The first is TayDDF.py which performs Data Driven Forecasting with the Taylor order only going up to 2nd order; it is the simpler and quicker of the two files. The MultiTayDDF.py is a more flexible version (also slower and more complex) which can handle any order of the Taylor expansion of the unknown differential equations of the system being studied. See the comments in the python scripts for more details. I also include a jupyter notebook I used to generate the data to test chaotic models on DDF.
And finally, I include a few pictures of a result from this method to show its efficacy.

DDF Basics: Here is a quick refresher on what DDF is doing and what the code is trying to accomplish. Starting with the dynamical equations, we have some system that has the following differential equations

dx(t)/dt = F(x(t))

We want to model the behavior of the observed variable x(t), but F(x(t)) is unkown to us. We can approximate the problem with the Euler Formula

x(n+1) = x(n) + dt*F(x(t))

Now we want a Function Representation for F(x(t)). We choose a representation the form to sum over all polynomial orders (the Taylor expansion):

f(x(t)) = sum_i(w_i*p_i(x(t))

We use these two equations above to write down a cost function to fit our coefficients in the Taylor Series Expansion:

Minimize sum_length [(x(n+1)-x(n)) - sum_i(w_i*p_i(x(t))]^2

Because the function representation is linear in the coefficients, we can rewrite the formula in terms of W*X where W are the weights, and X is the value of the polynomial. This minimization problem can be solved with Ridge Regression:

W = YX^T(XX^T)^-1

[Y] = 1 x Time

[X] = Parameter Length x Time

with the minizimation done, f(x(t)) can now be used to forecast forward in time.
