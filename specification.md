# Specification

## Glossary
by value -> parameter ownership moved into method
by ref -> method takes reference of parameter
in place -> parameter is passed by mutable reference

## Polynomial
### Type
Real coefficients
### Creation
- Creation from real coefficients
  - slice
  - iterator
- Creation from real roots
  - slice
  - iterator
### Indexing
Coefficients indexing and manipulation
### Properties
- Degree calculation (Option type, None for zero polynomial) (by ref)
- Extension of the polynomial with high degree zero terms (in place)
- Monic representation and leading coefficient (in place, by ref)
- Zero polynomial (additive identity)
- One polynomial (multiplicative identity)
- Evaluation (real and complex numbers) (by value, by ref)
### Roots
- Real roots calculation (Option type) (eigenvalues method)
- Complex roots calculation (eigenvalues and iterative method)
- Polynomial derivative (by ref)
- Polynomial integral (by ref)
### Arithmetic operations
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
### Formatting
Polynomial formatting

## Polynomial matrices

- Formatting the matrix

## Units of measurement
### Units
- Decibel
- Seconds (time)
- Hertz (frequency)
- Radians per second (angular frequency)
### Conversions
- Conversion Hertz <-> Radians per second
- Conversion Hertz <-> seconds
- Unit formatting as floating point and exponential form

## Utilities
- Natural pulse of a complex number
- Damp of a complex number
- Zip two iterators with the given function
- Zip two iterators with the given function extending the shorter

## Linear system - state space representation
#### Type
- Generic system (it can be used for continuous and discrete time)
- Create system given the dimension and the matrices as row major slices
#### Properties
- Dimensions of the system (number of inputs, states and outputs)
- Complex poles of the system
- Equilibrium points (x and y coordinates, formatting)
- Controllability matrix
- Observability matrix
- Formatting of the system
#### Conversions
- From single input single output state space representation to transfer function
- From multiple input multiple output state space representation to a matrix of transfer functions
### Continuous linear system state space representation
#### Properties
- Calculate equilibrium point
- System stability
#### System evoution solvers
- Time evolution of the system using Runge-Kutta second order method, returning an iterator
- Time evolution of the system using Runge-Kutta fourth order method, returning an iterator
- Time evolution of the system using Runge-Kutta-Fehlberg 45 with adaptive step method, returning an iterator
- Time evolution of the system using Radau of order 3 with 2 step method, returning an iterator
- Discretization of the system with forward Euler, backward Euler and Tustin methods
### Discrete linear system state space representation
#### Properties
- Calculate equilibrium point
- System stability
#### Time evolution
- Time evolution of the system given an input function, returning an iterator
- Time evolution of the system with input values supplied by an iterator, returning an iterator

## Continuous system solver (integrators)
- Runge-Kutta 2nd and 4th order iterator (current time, state and output)
- Runge-Kutta-Fehlberg 45 iterator (current time, state, output and maximum absolute error)
- Radau order 3 with 2 steps iterator (current time, state and output)

## Transfer function
#### Type
Generic transfer function
#### Properties
- Create from two polynomial (non zero numerator and denominator)
- Reference to the numerator and the denominator
- Calculate real/complex poles and zeros
- Normalization of the transfer function (monic denominator) (by ref, in place)
- Evaluation of a transfer function with complex or real numbers
- Formatting of a transfer function
#### Arithmetic operations
- Reciprocal of a transfer function (in place, by ref and by value)
- Negation of a transfer function (by ref, by value)
- Addition between transfer functions (by ref, by value)
- Subtraction between transfer functions (by ref, by value)
- Multiplication between transfer functions (by ref, by value)
- Division between transfer functions (by ref, by value)
#### Feedback
- Negative unit feedback
- Positive unit feedback
#### Conversions
From transfer function to state space representation
### Continuous time transfer function
#### Properties
- Time delay
- Static gain
- Initial value given a unitary step input
- Initial value derivative given a unitary step input
- Sensitivity function
- Complementary sensitivity function
- Sensitivity to control function
- Root locus for a given feedback gain
#### Plots
- Root locus plot
- Bode plot
- Polar plot
#### Discretization
Discretization of a continuous time transfer function (forward Euler, backward Euler and Tustin metods)
### Discrete time transfer function
#### Properties
- Time delay
- Static gain
- Initial value given a unitary step input
#### Autoregressive moving average representation
- Autoregressive moving average representation given an input function
- Autoregressive moving average representation with input values supplied by an iterator
### Matrix of transfer functions
#### Properties
- Retrieve the characteristic polynomial of the system
- Evaluation of the matrix give a vector of complex numbers
- Indexing (mutable) of the numerator of the matrix
- Formatting of the matrix

## Plots
### Bode plot
- Plot as an iterator of magnitude and radians
- Conversion of values into decibels and degrees
- Get angular frequency or frequency, magnitude and phase
### Polar plot
- Plot as an iterator of complex numbers
- Get real part, imaginary part, magnitude and phase
### Root locus plot
Plot as an iterator of transfer constant and roots

## Proportional integral derivative controller (PID)
- Create ideal PID
- Create real PID
- Generate the transfer function for the given PID

## Signals
### Continuous signals
- Zero input
- Step input
- Sine input
### Discrete signals
- Zero input
- Step input
- Impulse input
