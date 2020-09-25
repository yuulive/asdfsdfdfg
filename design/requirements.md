# Specific requirements

## External interfaces

### Data structures

This library must present as interface structures of the `Rust` standard library, or structure created by this library.

Using as interface as structure defined in a dependency exposes the internal implementation ad forces the use of that dependency.

### Numbers and units of measurement

Where the quantity and the related unit of measurement is identifiable, the latter shall be used as interface in place of primitive numeric types.

## Functional requirements

### Polynomials

#### Polynomial creation

FR1.1 The user shall be able to create polynomials with real coefficients, supplying the coefficients of the roots, both as list or as iterator.

FR1.2 In particular the coefficients shall be supplied from the lowest to the highest degree monomial. It is necessary to put zeros where the monomial is null.

FR1.3 Both the null polynomial and the unity polynomial can be created.

#### Indexing

FR1.4 The coefficients can be indexed, the index of the monomial is equal to its degree. The indexed coefficient shall be modifiable.

#### Properties

It is possible to operate on same polynomial properties:

FR1.5 Degree calculation (it is undefined for null polynomial) (by ref);

FR1.6 Extension of the polynomial with zeros in the highest degree monomials (in place);

FR1.7 The transformation into a monic polynomial and the return of the leading coefficient (in place, by ref);

FR1.8 Evaluation of a polynomial both with real and complex numbers using Horner's method (by value, by ref);

FR1.9 Rounding towards zero of the coefficients given an absolute tolerance (by ref, in place).

#### Roots

FR1.10 It is possible to calculate the roots of a polynomial. In the case of complex roots both the eigenvalues and iterative [3, 4, 5, 6, 8, 9] methods are available. In the case the user needs real roots, a result is supplied (with eigenvalues method) only in the case that all roots are real.

#### Arithmetical operations and infinitesimal calculus

FR1.11 On polynomial it is possible to perform arithmetical operations, both between polynomials and scalars, and operations of infinitesimal calculus:
- negation of the polynomial (by value, by ref)
- addition, subtraction, division and division reminder [7] between polynomials (by value, by ref)
- multiplication between polynomials both with the convolution method and the fast Fourier transform method [8] (by value, by ref)
- addition, subtraction, multiplication, and division with a scalar (by value, by ref)
- calculation of the derivative and the integral of the polynomial (by ref)
- evaluation of polynomial ratios avoiding overflows (by ref)

#### Formatting

FR1.12 It is available a standard formatting for the output of the polynomial as string.

#### Polynomial matrices

FR1.13 The matrices of polynomials cannot be directly created by the user, they are considered an internal implementation and they are not externally accessible. It is possible to index them and format them for the output as strings

### Units of measurement

#### Units

FR2.1 The library defines the following units of measurement, whose floating point value can be publicly accessed:
- decibel
- seconds (time)
- Hertz (frequency)
- radians per second (angular frequency)

FR2.2 The formatting of units is done as floating point numbers and in exponential form.

#### Conversions

FR2.3 The following conversions between units of measurement are available:
- conversion Hertz - radians per second
- inversion Hertz - seconds

### Utilities

#### Pulse and damp

FR3.1 Given a complex number it is possible to calculate the natural pulse and the damp [1].

### Time invariant linear system - state representation [1]

#### Linear system creation

FR4.1 It is possible to create linear systems both continuous and discrete time, it is necessary to insert the dimensions of the system and supply every matrix as a vector with elements row-wise.

FR4.2 The linear system can be created by the realization of a transfer function.

#### Properties

FR4.3 The user shall be able to obtain the properties of the linear system, i.e. the dimensions of the system (number of inputs, states and outputs), the complex poles of the system, the equilibrium points, the controllability matrix, the observability matrix. It shall be possible to determine if the system is stable.

#### Formatting

FR4.4 It is available a standard formatting for the output of the linear system as string.

#### Continuous time system

FR4.5 For continuous time systems shall be available integrator for the evolution of the system with time [2]:
- Runge-Kutta method of second and fourth order, returns an iterator;
- Runge-Kutta-Fehlberg method of order 4/5 with adaptive steps, returns and iterator;
- Radau method of order 3 with 2 steps, returns an iterator.

FR4.6 It shall be possible to discretize a system using forward Euler, backward Euler and Tustin methods.

#### Discrete time system

FR4.7 For discrete time systems it shall be possible to determine the evolution with time of the system given an input function or given an input supplied by and iterator, returns an iterator.

### Transfer functions [1]

#### Creation

FR5.1 The user shall be able to create transfer functions given two polynomials for the numerator and the denominator, for continuous and discrete time systems, it is not possible to use null polynomials for the numerator and the denominator.

FR5.2 It is possible to create a transfer function with a conversion from a linear system SISO [10].

FR3 It is possible to define a time delay function, both continuous and discrete time.

#### Properties

FR5.4 From the transfer function it is possible to extract is properties, i.e. the calculation of poles and zeros, both real and complex, the evaluation of the transfer function with real and complex numbers, the determination of the static gain and the initial value as response to a unity step.

#### Manipulation

FR5.5 It shall be possible to obtain a reference to the numerator and the denominator, normalise the transfer function, with a monic denominator (by ref, in place).

FR5.6 From a transfer function it is possible to obtain the transfer function of the system with a unity negative or positive feedback.

#### Formatting

FR5.7 It is available a standard formatting for the transfer function as string.

#### Arithmetical operations

FR5.8 It is possible to perform the following arithmetical operations on transfer functions:
- reciprocal of a transfer function (in place, by ref and by value)
- negation of a transfer function (by ref, by value)
- addition between transfer functions (by ref, by value)
- subtraction between transfer functions (by ref, by value)
- multiplication between transfer functions (by ref, by value)
- division between transfer functions (by ref, by value)

#### Continuous time

FR5.9 For the continuous time transfer functions it is possible to calculate the initial value of the derivative in response to a unity step, the root locus for a given gain in feedback and determine, given a controller, the sensitivity, the complementary sensitivity and the control sensitivity functions.

FR5.10 It is possible to generate the points for the Bode diagram, the polar diagram and the root locus diagram of the transfer function.

FR5.11 It shall be possible to discretize a continuous time transfer function with forward Euler, backward Euler and Tustin methods.

#### Discrete time

FR5.12 Given a discrete time transfer function it is possible to create an autoregressive moving average representation with a function as input or with an iterator input, in order to determine the time evolution of the system.

#### Matrix of transfer functions

FR5.13 From a time invariant MIMO linear system it is possible to create a matrix of transfer functions [10]. From this matrix it is possible to extract the characteristic polynomial common to every transfer functions, evaluate the matrix given a vector of complex numbers and index the numerators of the elements of the matrix.

### Diagrams [1]

#### Bode diagram

FR6.1 The user can obtain the points of the Bode diagram, the points are returned as iterators of the modulus and the phase, it is available the conversion of the values into decibel and degrees.

#### Polar diagram

FR6.2 The user can obtain the points of the polar diagram, the points are returned as an iterator of complex numbers.

#### Root locus diagram

FR6.3 The user can obtain the points of the root locus diagram, the points are returned as an iterator of the transfer constant and the roots.

### Proportional-integral-derivative controller (PID) [1]

FR7.1 The user can create ideal and real PID controllers, from the controllers it is possible to generate the transfer function of the given PID.

### Signals [1]

#### Continuous signals

FR8.1 Common continuous time signals are defined:
- null signal
- impulse signal
- step signal
- sinusoidal signal

#### Discrete signals

FR8.2 Common discrete time signals are defined:
- null signal
- impulse signal
- step signal
