# Test Plan

## Black box testing

Black box tests verify the properties of the mathematical entities defined in the library.

Tests have defined the path in the test files, the inputs and the expected outputs.

### Polynomials

TC1.1 A polynomial multiplied by one returns the same polynomial.
> tests/polynomial/multiplicative_unity

- input: p1 (arbitrary polynomial, null polynomial), p2 (unity polynomial, scalar 1)
- output: p1

TC1.2 A polynomial multiplied by zero returns zero.
> tests/polynomial/multiplicative_null

- input: p1 (arbitrary polynomial, null polynomial), p2 (null polynomial, scalar 0)
- output: null polynomial

TC1.3 If a non-null polynomial is multiplied by another polynomial and divided by the same, the result is equal to the first polynomial.
> tests/polynomial/multiplicative_inverse

- input: p1 (arbitrary polynomial), p2 (arbitrary polynomial)
- output: p1

TC1.4 A polynomial plus or minus zero is equal the the polynomial itself.
> tests/polynomial/additive_invariant

- input: p1 (arbitrary polynomial), p2 (null polynomial, scalar 0)
- output: p1

TC1.5 If to a polynomial is added and subtracted another polynomial, the result is equal to the first polynomial.
> tests/polynomial/additive_inverse

- input: p1 (arbitrary polynomial), p2 (arbitrary polynomial)
- output: p1

TC1.6 The number of roots of a polynomial is equal to its degree.
> tests/polynomial/roots_degree

- input: 1, 1+2x, 1+2x+3x^{2}
- output: Some(0), Some(1), Some(2)

TC1.7 A null polynomial does not have a defined degree.
> tests/polynomial/no_degree

- input: null polynomial
- output: None

TC1.8 The derivative of a polynomial has one less degree than the starting polynomial.
> tests/polynomial/derivation

- input: 2x+3x^{2}, 2+6x, 6
- output: Some(1), Some(0), None

TC1.9 The integral of a polynomial has one more degree than the starting polynomial.
> tests/polynomial/integration

- input: 0, -1+x, 2-x+\frac{x^{2}}{2}
- output: Some(0), Some(1), Some(2)

TC1.10 The stationary points of a polynomial function are the roots of its derivative.
> tests/polynomial/maximum_minimum

- input: \left(1+x\right)x\left(-1+x\right)
- output: stationary points in x=\pm0.57735

TC1.11 The iterative root finding method must be stable for nearly multiple zeros ([16] page 6).
> tests/polynomial/nearly_multiple_zeros

- input: P4, P5, P6, P8
- output: roots must all be real, from root it is possible to recover the original polynomial with given error or roots must be within given error.

### Linear systems

TC2.1 The initial value of the output of the system in response to a step is the limit to infinity of its transfer function ([1], para. 5.4.1).
> tests/linear_system/inital_value

- input: \frac{1+5s}{\left(1+2s\right)\left(1+s\right)}
- output: \lim_{x\to\infty}\frac{1+5x}{\left(1+2s\right)\left(1+s\right)}=0

TC2.2 A linear system can be represented with state variables or with a transfer function. A SISO system can be transformed from the state representation to the transfer function. ([1], para. 5.5.3).
> tests/conversions/from_tf_to_ss

- input: \frac{4}{4+s+2s^{2}}
- output: A=\left[\begin{array}{cc}
0 & -2\\
1 & -0.5
\end{array}\right], B=\left[\begin{array}{c}
2\\
0
\end{array}\right], C=\left[\begin{array}{cc}
0 & 1\end{array}\right], D=\left[0\right]

TC2.3 The poles of a system are equal to the eigenvalues of the A matrix ([1], para. 3.2.5).
> tests/linear_system/poles_eigenvalues

- input: \frac{4}{\left(1+s\right)\left(2+s\right)\left(3+s\right)}, transfer function realization
- output: -1, -2, -3

TC2.4 The response of an asymptotically stable system, after enough time, is independent from the initial state ([1], para. 3.4.6).
> tests/linear_system/initial_state_independence

- input: discrete time A=\left[\begin{array}{cc}
0.3 & 0\\
0 & 0.25
\end{array}\right], B=\left[\begin{array}{c}
3\\
-1
\end{array}\right], C=\left[\begin{array}{cc}
1 & 1\end{array}\right], D=\left[1\right], initial state 1: \left[\begin{array}{c}
0\\
0
\end{array}\right], initial state 2: \left[\begin{array}{c}
1\\
1
\end{array}\right]
- output: responses after 30 steps must be equal to the static gain

TC2.5 In an asymptotically stable system, the movement of the state and output as response to a time limited input goes to zero ([1], para. 3.4.6). As response to an impulse, both the state and the output go to zero ([1], para. 3.4.6).
> tests/linear_system/to_zero

- input: \frac{1+5s}{\left(1+2s\right)\left(1+s\right)}, realization of transfer function
- output: 0

TC2.6 The series connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable ([1], para. 6.4.1).
> tests/linear_system/series_system

- input: G_{1}=\frac{1}{\left(0.7+z\right)\left(0.5+z\right)}, G_{2}=\frac{1}{\left(0.2+z\right)\left(0.25+z\right)}, G_{3}=\frac{1}{\left(-2+z\right)\left(0.25+z\right)}
- output: G_{1}\cdot G_{2} is stable, G_{1}\cdot G_{3} is unstable

TC2.7 The parallel connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable ([1], para. 6.4.2).
> tests/linear_system/parallel_system

- input: G_{1}=\frac{1}{\left(1+s\right)\left(0.5+s\right)}, G_{2}=\frac{1}{\left(2+s\right)\left(0.25+s\right)}, G_{3}=\frac{1}{\left(-2+s\right)\left(0.25+s\right)}
- output: G_{1}+G_{2} is stable, G_{1}+G_{3} is unstable

### Continuous time

TC3.1 A linear system whose eigenvalues have negative real part is stable ([1], para. 3.4.3).
> tests/continuous_linear_system/stability

- input: G_{1}=\frac{0.5+1.5s}{\left(1+s\right)\left(1.3+s\right)\left(15+s\right)}, G_{2}=\frac{0.5+1.5s}{\left(1+s\right)\left(-0.3+s\right)\left(5+s\right)}
- output: G_{1} is stable, G_{2} is unstable

TC3.2 A system without null eigenvalues has a unique equilibrium state ([1], para. 3.3, 3.4.6).
> tests/continuous_linear_system/equilibrium

- input: A=\left[\begin{array}{cc}
0 & 1\\
-1 & -1
\end{array}\right], B=\left[\begin{array}{c}
0\\
1
\end{array}\right], C=\left[\begin{array}{cc}
0 & 1\end{array}\right], D=\left[0\right], input=\left[1\right]
- output: state=\left[\begin{array}{c}
1\\
0
\end{array}\right], output=\left[0\right]

TC3.3 A system with at least one null eigenvalue has none or infinite equilibrium states ([1], para. 3.3).
> tests/continuous_linear_system/no_equilibrium

- input: A=\left[\begin{array}{cc}
0 & 0\\
1 & 2
\end{array}\right], B=\left[\begin{array}{cc}
1 & 2\\
3 & 4
\end{array}\right], C=\left[\begin{array}{cc}
5 & 6\end{array}\right], D=\left[\begin{array}{cc}
0 & 0\end{array}\right], input=\left[\begin{array}{c}
1\\
1
\end{array}\right]
- output: None

TC3.4 The static gain of an asymptotically stable system in response the a step is equal to the value of its transfer function in zero ([1], para. 5.3.1).
> tests/continuous_linear_system/static_gain

- input: G(s)=\frac{1+5s}{\left(1+2s\right)\left(1+s\right)}, realization of transfer function
- output: G(0)=1=output after 15 seconds

### Discrete time

TC4.1 A linear system whose eigenvalues have modulus less then one is stable ([1], para. 8.5.3).
> tests/discrete_linear_system/stability

- input: G_{1}=\frac{0.5+1.5z}{\left(0.3+z\right)\left(0+z\right)\left(0.99+z\right)}, G_{2}=\frac{0.5+1.5z}{\left(1+z\right)\left(-0.3+z\right)\left(5+z\right)}
- output: G_{1} is stable, G_{2} is unstable

TC4.2 A system without eigenvalues with modulus equal to one has a unique equilibrium state ([1], para. 8.4.7).
> tests/discrete_linear_system/equilibrium

- input: A=\left[\begin{array}{cc}
0.6 & 0\\
0 & 0.4
\end{array}\right], B=\left[\begin{array}{c}
1\\
5
\end{array}\right], C=\left[\begin{array}{cc}
1 & 3\end{array}\right], D=\left[0\right], input=\left[1\right]
- output: state=\left[\begin{array}{c}
2.5\\
8.\bar{3}
\end{array}\right], output=\left[27.5\right]

TC4.3 A system with at least on eigenvalue with modulus equal to one has none of infinite equilibrium states ([1], para. 8.4.7).
> tests/discrete_linear_system/no_equilibrium

- input: A=\left[\begin{array}{cc}
0.6 & 0\\
0 & 1
\end{array}\right], B=\left[\begin{array}{c}
1\\
5
\end{array}\right], C=\left[\begin{array}{cc}
1 & 3\end{array}\right], D=\left[0\right], input=\left[1\right]
- output: None

TC4.4 The static gain of an asymptotically stable system in response the a step is equal to the value of its transfer function in one ([1], para. 9.2.5).
> tests/discrete_linear_system/static_gain

- input: G(z)=\frac{-0.5}{\left(-0.5+z\right)\left(-0.5+z\right)}, realization of transfer function
- output: G(1)=-2=output after 30 steps

TC4.5 The response of a Finite Impulse Response (FIR) system to an impulse is zero after an number of steps equal to its order ([1], para. 9.3.5).
> tests/discrete_linear_system/fir_impulse

- input: G(z)=1.016\frac{0.015+0.031z+0.063z^{2}+0.125z^{3}+0.25z^{4}+0.5z^{5}}{z^{6}}
- output: output after 6 steps = 0

TC4.6 The response of a Finite Impulse Response (FIR) system to a step is equal to its static gain after a number of steps equal to is order ([1], para. 9.3.5).
> tests/discrete_linear_system/fir_step

- input: G(z)=1.016\frac{0.015+0.031z+0.063z^{2}+0.125z^{3}+0.25z^{4}+0.5z^{5}}{z^{6}}
- output: output after 6 steps = G(1)=1

TC4.7 The moving average of values in time is representable with a transfer function Finite Impulse Response (FIR) [1].
> tests/discrete_transfer_function/left_moving_average

- input: y\left[k\right]=\frac{u\left[k\right]+u\left[k-1\right]+u\left[k-2\right]}{3}\rightarrow G(z)=\frac{1+z+z^{2}}{3z^{2}}, values =[9.,8.,9.,12.,9.,12.,11.,7.,13.,9.,11.,10.]
- output: [x,x,8.667,9.667,10.,11.,10.667,10.,10.333,9.667,11.,10.]

### Diagrams

TC5.1 The Bode diagram of a system with two conjugate complex poles must have a resonance peak ([1], para. 5.4.4).
> tests/plots/bode_plot

- input: G(s)=\frac{1}{1+2\frac{\xi s}{\omega_{n}}+\frac{s^{2}}{\omega_{n}^{2}}}, \xi=0.1, \omega_{n}=1
- output: \left|G(j\omega_{n})\right|_{dB}=\left|\frac{1}{2\left|\xi\right|}\right|_{dB}=13.9794

TC5.2 A transfer function of a minimum phase system generates a polar diagram whose points strictly decrease with frequency increase ([1], para. 7.7).
> tests/plots/polar_plot

- input: G(s)=\frac{5}{\left(1+s\right)\left(10+sx\right)}
- output: \left|G(j\omega)\right| con 0.1\le\omega\le10 is strictly decreasing

TC5.3 The number of branches of the direct root locus is equal to the degree of the denominator, the root locus is symmetric with respect to the real axis ([1], para. 13.2.2).
> tests/plots/root_locus_plot

- input: G(s)=\frac{k}{s\left(3+s\right)\left(5+s\right)}
- output: 3 branches, symmetric, limit value for stability =120

## White box testing

Every module must contain a test submodule, whose purpose it to test all the functions present inside the module.

This tests are dependent from the implementation of the method, it is not given a detailed description, but the requirements are listed and what the test shall verify. More information must be added in the source code.

### Module controllers

#### pid

Creation of a real and ideal PID controller and generation of the related transfer function.

### Module linear_system

Creation of the structure that contains the dimensions of a system and check of the correct initialization of a system.

Calculation of the eigenvalues of a system of first, second and third order.

Le Verrier algorithm with single and double precision floating point numbers, both on a 1x1 matrix and higher dimensions.

Conversion from a transfer function to a representation with state variables, both through the canonical observability and controllability form.

Calculation of the observability and controllability matrices.

#### continuous

Determination of an equilibrium point and the evaluation of the stability of a system.

Evolution of a system using Runge-Kutta method of order 2 and 4, the Runge-Kutta-Fehlberg method of order 4/5 and the Radau method.

#### discrete

Determination of and equilibrium point and the evaluation of the stability of a system.

Evolution of a system with input supplied as a function or an iterator.

Discretization of a continuous time system using forward Euler, backward Euler and Tustin methods. Backward Euler and Tustin methods can fail.

#### solver

Creation of the data structures that are returned by the integrators.

### Module plots

#### bode

Creation of the iterator that supplies the diagram points, both as absolute values / Hertz and decibel / radians per second and counting of supplied points.

Creation of the structure that represents the points of the diagram.

#### polar

Creation of the iterator that supplies the diagram points and counting of supplied points.

Creation of the structure that represents the points of the diagram.

#### root_locus

Correctness of the parameters for the creation of the data structure of the diagram.

### Module polynomial

Formatting polynomial as string.

Creation from coefficient and from roots, supplied both as a list and as a iterator, creation of the null and unity polynomial.

Return coefficients as a list.

Calculation of the length and the degree of the polynomial.

Extension with zeros of the high degree coefficients.

Evaluation of the polynomial both with real and complex numbers.

Indexing of polynomial coefficients.

Derivative and integral of the polynomial.

Transformation into a monic polynomial and return of the leading coefficient.

Rounding towards zero of the coefficients.

Creation of the companion matrix.

Calculation of the real and complex roots of a polynomial of first, second and third degree, both with the method of eigenvalues and the iterative method. The null roots shall be removed from the calculation.

#### arithmetic

Arithmetic operations defined between polynomial and with real numbers.

Multiplication through fast Fourier transformation.

#### fft

Bit permutation of a integer number.

Calculation of the direct and inverse fast Fourier transformation.

#### roots

Calculation of the roots with iterative method.

### Module polynomial_matrix

#### PolyMatrix

Creation of a polynomial whose elements are matrices, eliminating null terms.

Multiplication coefficient by coefficient times a matrix both on the right and on the left.

Evaluation supplying a matrix of values.

Addition between polynomials of matrices.

#### MatrixOfPoly

Creation of a matrix whose elements are polynomial, that can be indexed.

If the matrix has dimension 1x1 it is possible to extract the single element.

### Module signals

#### continuous

Creation and output of a null, impulse, step and sinusoidal signal.

#### discrete

Creation and output of a null, impulse and step signal.

### Module transfer_function

Creation of a transfer function.

Evaluation of a transfer function given a complex number.

Inversion of a function and arithmetical operations of negation, addition, subtraction, multiplication and division.

Calculation of poles and zeros, both real and complex.

Determination of the system given a function with a unitary positive or negative feedback.

Formatting of the function as a string.

Normalization of the function with a monic polynomial as denominator.

#### continuous

Creation of a delay function.

Calculation of the static gain.

Determination of the stability of a system.

Creation of a Bode diagram, polar diagram and a root locus diagram.

Calculation of the initial value of the system evolution and its first derivative.

Calculation of the sensitivity, complementary sensitivity and control sensitivity functions.

#### discrete

Creation of a transfer function of a discrete time system and a delay function.

Calculation of the initial value of the system evolution.

Calculation of the static gain.

Determination of the stability of a system.

Evaluation of the function given a complex number.

Creation of an autoregressive moving average form.

#### discretization

Creation of a discrete time transfer function from a continuous time one.

Check of the correct discretization with forward Euler, backward Euler and Tustin methods.

#### matrix

Creation of a matrix of transfer functions.

Evaluation of the matrix given a vector of complex numbers.

Indexing and formatting as string of the matrix.

### Module units

Creation of decibel from floating point numbers.

Conversion between Hertz and radians per second. Inversion between seconds and Hertz.

Formatting of units of measurement as strings.

### Module utils

Determination of the pulse and the dump of a complex number.

Zip two iterators with different lengths, also through a function, at the shortest elements are added.

## Documentation tests

Every public method of the library shall have a documentation in the source code, in addition there shall be an example of the use of the method.

This example it will compiled and executed during the test phase, ensuring that the documentation remains aligned with the code modifications.

## Test coverage

The test development shall tend to obtain the maximum line coverage.

The program to determine the test coverage is `tarpaulin` [15], it is executed during the continuous integration phases.
