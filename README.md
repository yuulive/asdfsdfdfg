# Automatica - Automatic Control Systems Library

[Documentation](https://daingun.gitlab.io/automatica)

## State-Space representation

Creation of state-space of a linear, time indepentdent system throught the four matrices A, B, C, D.  
Calculate the poles of the system.  
Calculate the equiliblium point (both state and output) of the system from the given input.

## Transfer function representation

### Sigle Input Single Output (SISO)
Creation of a single transfer function give a polynomial numerator and denominator.  
Calculate the (complex) poles and (complex) zeros of the function.  
Evaluation of the transfer function at the given input.  

### Multiple Input Multiple Output (MIMO)
Creation of a matrix of transfer functions, given a matrix of polynomials and the characteristic polynomial.  
Evaluation of the matrix at the given vector of inputs.  
(Mutable) Indexing of the matrix elements numerators.  

## Conversion between representations

SISO state-space -> transfer function  
MIMO state-space -> matrix of transfer functions  
Transfer function -> state-space (observable form)

## Plots

### Bode

Calculate the magnitude and phase for a single transfer function in an interval of frequencies.

### Polar

Polar plot of a transfer function.

## Controllers

PID (Proportional-integral-derivative) controller, both ideal and real.

## Polynomials

Polynomial creation from coefficients or roots.  
Polynomial evaluation with Horner method.  
(Mutable) Indexing of polynomial coefficients.  
Polynomials addition, subtraction and multiplication.  
Polynomial and scalar addition, subtraction, multiplication and division.  
Polynomial roots finding (real and complex).  
Creation of a matrix of polynomials.

## Examples

Examples of library usage can be found in the examples/ folder.
