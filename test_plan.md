# Test Plan

## Black box testing

### Polynomial
A polynomial multiplied by one returns the same polynomial.

A polynomial multiplied by zero returns zero.

Zero plus a polynomial is equal to the polynomial.

If a polynomial is added and subtracted to another polynomial the result is equal to the latter.

If a non zero polynomial is multiplied and divided to another polynomial the result equals to the latter.

A polynomial must have the same number of roots as its degree.

A zero polynomial has no defined degree.

### Linear System

The initial value of a system is the limit to infinity of its transfer function.

A linear system can be represented both with state-space representation of with transfer function.

A single input single output system can be transformed between state-representation and transfer function forms.

The poles of a system are equal to the eigenvalues of the A matrix.

An asymptotically stable system response after enough time is independent from the initial state. (3.4.6)

The response to an impulse of the state and the output tends to zero. (3.4.6)

The response of the state and the output to an input limited in time tends to zero. (3.4.6)

Series connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable.

Parallel connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable.

#### Continuous time

A linear system whose eigenvalues have negative real part is stable. (3.4.3)

A system with no null eigenvalues has a unique equilibrium state. (3.3, 3.4.6)

A system with null eigenvalues has a none or infinite equilibrium states. (3.3)

An asymptotically stable transfer function gain for a step input is the value at zero.

#### Discrete time

A linear system whose eigenvalues have modulus less than one is stable. (8.5.3)

A system with no eigenvalues with modulus equal to one has a unique equilibrium state. (8.4.7)

A system with eigenvalues with modulus equal to one has a none or infinite equilibrium states. (8.4.7)

An asymptotically stable transfer function gain for a step input is the value at one. (9.2.5)

A Finite Impulse Response (FIR) system output response to an impulse is zero after a number of steps equal to its order. (9.3.5)

A Finite Impulse Response (FIR) system output response to a step is equal to its gain after a number of steps equal to its order. (9.3.5)
