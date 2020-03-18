# Glossary
by value -> parameter ownership moved into method
by ref -> method takes reference of parameter
in place -> parameter is passed by mutable reference

# Polynomial

- Real coefficients
- Creation from a slice of real coefficients
- Creation from a slice of real roots
- Coefficients indexing and manipulation
- Degree calculation (Option type, None for zero polynomial) (by ref)
- Extension of the polynomial with high degree zero terms (in place)
- Real roots calculation (Option type) (eigenvalues method)
- Complex roots calculation (eigenvalues and iterative method)
- Monic representation and leading coefficient (in place, by ref)
- Polynomial derivative (by ref)
- Polynomial integral (by ref)
- Evaluation (scalar and complex numbers) (by value, by ref)
- Negation of polynomial (by value, by ref)
- Addition between polynomials (by value, by ref)
- Subtraction between polynomials  (by value, by ref)
- Multiplication between polynomials (convolution and fast fourier transform) (by value, by ref)
- Division between polynomials (by value, by ref)
- Remainder between polynomials (by value, by ref)
- Addition with scalar (commutative) (by value, by ref)
- Subtraction with scalar (commutative) (by value, by ref)
- Multiplication with scalar (commutative) (by value, by ref)
- Division with scalar (by value, by ref)
- Zero polynomial (additive identity)
- One polynomial (multiplicative identity)
- Polynomial formatting

# Polynomial matrices

- Formatting the matrix

# Units of measurement

- Decibel
- Seconds (time)
- Hertz (frequency)
- Radians per second (angular frequency)
- Conversion Hertz <-> Radians per second
- Conversion Hertz <-> seconds
- Unit formatting as floating point and exponential form

# Utilities

- Natural pulse of a complex number
- Damp of a complex number
- Zip two iterators / slices with the given function
- Zip two iterators / slices with the given function extending the shorter

# Linear system - state space representation

- Generic system
- Dimensions of the system (number of inputs, states and outputs)
- Create system given the dimension and the matrices as row major slices
- Complex poles of the system
- Controllability matrix
- Observability matrix
- Conversion from transfer function to state space representation
- Formatting of the system
- Equilibrium points (x and y coordinates, formatting)

## Continuous linear system state space representation

- Calculate equilibrium point
- Determine if the system is stable
- Time evolution of the system using Runge-Kutta second order method, returning an iterator
- Time evolution of the system using Runge-Kutta fourth order method, returning an iterator
- Time evolution of the system using Runge-Kutta-Fehlberg 45 with adaptive step method, returning an iterator
- Time evolution of the system using Radau of order 3 with 2 step method, returning an iterator

## Discrete linear system state space representation

- Calculate equilibrium point
- Time evolution of the system given an input function, returning an iterator
- Time evolution of the system given an iterator with input values, returning an iterator
- Determine if the system is stable
- Discretization of a continuous system with forward Euler, backward Euler and Tustin methods
- Time evolution struct (current time, state and output)

## Continuous system solver (integrators)

- Runge-Kutta 2nd and 4th order iterator (current time, state and output)
- Runge-Kutta-Fehlberg 45 iterator (current time, state, output and maximum absolute error)
- Radau order 3 with 2 steps iterator (current time, state and output)

# Transfer function

- Generic transfer function
- Create from two polynomial (non zero numerator and denominator)
- Reference to the numerator and the denominator
- Reciprocal of a transfer function (in place, by ref and by value)
- Calculate real/complex poles and zeros
- Negative unit feedback
- Positive unit feedback
- Normalization of the transfer function (monic denominator) (by ref, in place)
- Conversion from single input single output state space representation to transfer function
- Negation of a transfer function (by ref, by value)
- Addition between transfer functions (by ref, by value)
- Subtraction between transfer functions (by ref, by value)
- Multiplication between transfer functions (by ref, by value)
- Division between transfer functions (by ref, by value)
- Evaluation of a transfer function with complex or real numbers
- Formatting of a transfer function

## Continuous time transfer function

- Time delay
- Initial value given a unitary step input
- Initial value derivative given a unitary step input
- Sensitivity function
- Complementary sensitivity function
- Sensitivity to control function
- Root locus for a given feedback gain
- Root locus plot
- Static gain
- Bode plot
- Polar plot

## Discrete time transfer function

- Time delay
- Initial value given a unitary step input
- Static gain
- Autoregressive moving average representation given an input function
- Autoregressive moving average representation given an interator with input values
- Discretization of a continuous time transfer function (forward Euler, backward Euler and Tustin metods)

## Matrix of transfer functions

- Retrieve the characteristic polynomial of the system
- Evaluation of the matrix give a vector of complex numbers
- Conversion from multiple input multiple output state space representation
- Indexing (mutable) of the numerator of the matrix
- Formatting of th matrix

# Plots

## Bode plot

- Plot as an iterator of magnitude and radians
- Conversion of values into decibels and degrees
- Get angular frequency or frequency, magnitude and phase

## Polar plot

- Plot as an iterator of complex numbers
- Get real part, imaginary part, magnitude and phase

## Root locus plot

- Plot as an iterator of transfer constant and roots

# Proportional integral derivative controller (PID)

- Create ideal PID
- Create real PID
- Generate the transfer function for the given PID

# Signals

## Continuous signals

- Zero input
- Step input
- Sine input

## Discrete signals

- Zero input
- Step input
- Impulse input
