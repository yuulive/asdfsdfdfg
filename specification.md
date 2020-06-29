# Software Requirements Specification

## Introduction
### Purpose
This SRS is intended to provide the reference for the high level description of software.

The intended audience of the SRS is those who need to use this library in their software or those who want to contribute to the development of this library.

### Scope
The name of the library is `automatica`.

The library will provide an infrastructure that provide calculation methods for computer-aided control system design.
This software is not indended to provide an interface with the final user.

The applications of this library should be low level (or system level) subroutines. The benefits shall be fast calculation done directly at system level.

### Definitions, acronyms and abbreviations
Polynomial ->
Linear system ->
State space representation ->
Transfer function ->
Signal ->
SISO -> single input single output
by value -> parameter ownership moved into method
by ref -> method takes reference of parameter
in place -> parameter is passed by mutable reference

### References
P. Bolzern, R. Scattolini, N. Schiavoni, Fondamenti di controlli automatici, McGraw-Hill Education, 2015

A. Quarteroni, R. Sacco, F Saleri, Numerical Mathematics, Springer, 2000

O. Aberth, Iteration Methods for Finding all Zeros of a Polynomial Simultaneously, Math. Comput. 27, 122 (1973) 339–344.

D. A. Bini, Numerical computation of polynomial zeros by means of Aberth’s method, Baltzer Journals, June 5, 1996

D. A. Bini, L. Robol, Solving secular and polynomial equations: A multiprecision algorithm, Journal of Computational and Applied Mathematics (2013)

W. S. Luk, Finding roots of real polynomial simultaneously by means of Bairstow's method, BIT 35 (1995), 001-003

Donald Ervin Knuth, The Art of Computer Programming: Seminumerical algorithms, Volume 2, third edition, section 4.6.1, Algorithm D: division of polynomials over a field.

### Overview

## Overall description
### Product perspective
Possible UML diagram
#### System interfaces
#### User interfaces
#### Hardware interfaces
Reference/link to Rust tier list
#### Software interfaces
List of dependencies:
- num-traits
  * Trait system for numeric types. It is used to allow the writing of generic numeric methods
  * Source: [https://crates.io/crates/num-traits](https://crates.io/crates/num-traits)
  * Version: 0.2
- num-complex
  * Complex numbers library
  * Source: [https://crates.io/crates/num-complex](https://crates.io/crates/num-complex)
  * Version: 0.2
- nalgebra
  * Linear algebra library. It provides matrices an methods for matrices arithmetics, matrices decomposition and eigenvalues finder
  * Source: [https://crates.io/crates/nalgebra](https://crates.io/crates/nalgebra)
  * Version: 0.18.0
- ndarray
  * Multidimensional arrays. Complementary method where nalgebra requirements on input types are too string
  * Source: [https://crates.io/crates/ndarray](https://crates.io/crates/ndarray)
  * Version: 0.12.1
- approx
  * Approximate equalities for floating point numbers
  * Source: [https://crates.io/crates/approx](https://crates.io/crates/approx)
  * Version: 0.3.2
Development dependencies:
- quickcheck
  * Randomized testing
  * Source: [https://crates.io/crates/quickcheck](https://crates.io/crates/quickcheck)
  * Version: 0.9
- quickcheck_macros
  * Randomized testing macros
  * Source: [https://crates.io/crates/quickcheck_macros](https://crates.io/crates/quickcheck_macros)
  * Version: 0.9
#### Communications interfaces
The library does not have network interfaces.
#### Memory
There are no limits on memory needed to use this library.
#### Operations
No requirements on operations are given for this library.
#### Site adaptation requirements
No site adaptation requirements are needed for this library.

### Product functions
- Polynomial library
  * arithmetic methods
  * root finding methods
  * polynomial evaluation
- Linear system library
  * state space and transfer function representation, with transformations
  * continuous and discrete time systems
  * system evolution
  * continuous time system discretization
- Plot library
  * Bode plot
  * Polar plot
  * Root locus plot

### User characteristics
User of this library are required to have knowledge of control system design.

### Constraints
There are no constraints in the use of this library.
On development side correctness should be the primary concern.

### Assumptions and dependencies
At the current state of the SRS there are no changes that can affect the requirements of the SRS.

### Apportioning of requirements
Future requirements may be defined on performance.

## Specific requirements

### Polynomial
#### Type
Real coefficients
#### Creation
- Creation from real coefficients
  - slice
  - iterator
- Creation from real roots
  - slice
  - iterator
#### Indexing
Coefficients indexing and manipulation
#### Properties
- Degree calculation (Option type, None for zero polynomial) (by ref)
- Extension of the polynomial with high degree zero terms (in place)
- Monic representation and leading coefficient (in place, by ref)
- Zero polynomial (additive identity)
- One polynomial (multiplicative identity)
- Evaluation (real and complex numbers) (by value, by ref)
- Round off to zero give an absolute tolerance (by ref, in place)
#### Roots
- Real roots calculation (Option type) (eigenvalues method)
- Complex roots calculation (eigenvalues and iterative method)
- Polynomial derivative (by ref)
- Polynomial integral (by ref)
#### Arithmetic operations
- Negation of polynomial (by value, by ref)
- Addition between polynomials (by value, by ref)
- Subtraction between polynomials  (by value, by ref)
- Multiplication between polynomials (convolution and fast Fourier transform) (by value, by ref)
- Division between polynomials (by value, by ref)
- Remainder between polynomials (by value, by ref)
- Addition with scalar (commutative) (by value, by ref)
- Subtraction with scalar (commutative) (by value, by ref)
- Multiplication with scalar (commutative) (by value, by ref)
- Division with scalar (by value, by ref)
#### Formatting
Polynomial formatting

### Polynomial matrices
- Formatting the matrix

### Units of measurement
#### Units
- Decibel
- Seconds (time)
- Hertz (frequency)
- Radians per second (angular frequency)
#### Conversions
- Conversion Hertz <-> Radians per second
- Conversion Hertz <-> seconds
- Unit formatting as floating point and exponential form

### Utilities
- Natural pulse of a complex number
- Damp of a complex number
- Zip two iterators with the given function
- Zip two iterators with the given function extending the shorter

### Linear system - state space representation
##### Type
- Generic system (it can be used for continuous and discrete time)
- Create system given the dimension and the matrices as row major slices
##### Properties
- Dimensions of the system (number of inputs, states and outputs)
- Complex poles of the system
- Equilibrium points (x and y coordinates, formatting)
- Controllability matrix
- Observability matrix
- Formatting of the system
##### Conversions
From transfer function to state space representation
#### Continuous linear system state space representation
##### Properties
- Calculate equilibrium point
- System stability
##### System evolution solvers
- Time evolution of the system using Runge-Kutta second order method, returning an iterator
- Time evolution of the system using Runge-Kutta fourth order method, returning an iterator
- Time evolution of the system using Runge-Kutta-Fehlberg 45 with adaptive step method, returning an iterator
- Time evolution of the system using Radau of order 3 with 2 step method, returning an iterator
- Discretization of the system with forward Euler, backward Euler and Tustin methods
#### Discrete linear system state space representation
##### Properties
- Calculate equilibrium point
- System stability
##### Time evolution
- Time evolution of the system given an input function, returning an iterator
- Time evolution of the system with input values supplied by an iterator, returning an iterator

### Continuous system solver (integrators)
- Runge-Kutta 2nd and 4th order iterator (current time, state and output)
- Runge-Kutta-Fehlberg 45 iterator (current time, state, output and maximum absolute error)
- Radau order 3 with 2 steps iterator (current time, state and output)

### Transfer function
##### Type
Generic transfer function
##### Properties
- Create from two polynomial (non zero numerator and denominator)
- Reference to the numerator and the denominator
- Calculate real/complex poles and zeros
- Normalization of the transfer function (monic denominator) (by ref, in place)
- Evaluation of a transfer function with complex or real numbers
- Formatting of a transfer function
##### Arithmetic operations
- Reciprocal of a transfer function (in place, by ref and by value)
- Negation of a transfer function (by ref, by value)
- Addition between transfer functions (by ref, by value)
- Subtraction between transfer functions (by ref, by value)
- Multiplication between transfer functions (by ref, by value)
- Division between transfer functions (by ref, by value)
##### Feedback
- Negative unit feedback
- Positive unit feedback
##### Conversions
- From single input single output state space representation to transfer function
- From multiple input multiple output state space representation to a matrix of transfer functions
#### Continuous time transfer function
##### Properties
- Time delay
- Static gain
- Initial value given a unitary step input
- Initial value derivative given a unitary step input
- Sensitivity function
- Complementary sensitivity function
- Sensitivity to control function
- Root locus for a given feedback gain
##### Plots
- Root locus plot
- Bode plot
- Polar plot
##### Discretization
Discretization of a continuous time transfer function (forward Euler, backward Euler and Tustin methods)
#### Discrete time transfer function
##### Properties
- Time delay
- Static gain
- Initial value given a unitary step input
##### Autoregressive moving average representation
- Autoregressive moving average representation given an input function
- Autoregressive moving average representation with input values supplied by an iterator
#### Matrix of transfer functions
##### Properties
- Retrieve the characteristic polynomial of the system
- Evaluation of the matrix given a vector of complex numbers
- Indexing (mutable) of the numerator of the matrix
- Formatting of the matrix

### Plots
#### Bode plot
- Plot as an iterator of magnitude and radians
- Conversion of values into decibels and degrees
- Get angular frequency or frequency, magnitude and phase
#### Polar plot
- Plot as an iterator of complex numbers
- Get real part, imaginary part, magnitude and phase
#### Root locus plot
Plot as an iterator of transfer constant and roots

### Proportional integral derivative controller (PID)
- Create ideal PID
- Create real PID
- Generate the transfer function for the given PID

### Signals
#### Continuous signals
- Zero input
- Step input
- Sine input
#### Discrete signals
- Zero input
- Step input
- Impulse input
