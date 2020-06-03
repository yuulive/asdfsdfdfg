# Test Plan

## Black box testing

### Polynomial
A polynomial multiplied by one returns the same polynomial.
> tests/polynomial/multiplicative_unity

A polynomial multiplied by zero returns zero.
> tests/polynomial/multiplicative_null

A polynomial plus or minus zero is equal to the polynomial.
> tests/polynomial/additive_invariant

If a polynomial is added and subtracted to another polynomial the result is equal to the latter.
> tests/polynomial/additive_inverse

If a non zero polynomial is multiplied and divided to another polynomial the result equals to the latter.
> tests/polynomial/multiplicative_inverse

A polynomial must have the same number of roots as its degree.
> tests/polynomial/roots_degree

A zero polynomial has no defined degree.
> tests/polynomial/no_degree

A polynomial derivative shall have one less degree of the polynomial.
> tests/polynomial/derivation

A polynomial integral shall have one more degree of the polynomial.
> tests/polynomial/integration

### Linear System

The initial value of a system is the limit to infinity of its transfer function.

A linear system can be represented both with state-space representation of with transfer function.
> tests/conversions/from_tf_to_ss

A single input single output system can be transformed between state-representation and transfer function forms.
> tests/conversions/from_tf_to_ss

The poles of a system are equal to the eigenvalues of the A matrix.
> tests/linear_system/poles_eigenvalues

An asymptotically stable system response after enough time is independent from the initial state. (3.4.6)
> tests/linear_system/initial_state_independence

The response of the state and the output to an input limited in time tends to zero. (3.4.6)
In response to an impulse, both the state and the output tends to zero. (3.4.6)
> tests/linear_system/to_zero

Series connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable.
> tests/linear_system/series_system

Parallel connection of subsystems is asymptotically stable if and only if the subsystems are asymptotically stable.
> tests/linear_system/parallel_system

#### Continuous time

A linear system whose eigenvalues have negative real part is stable. (3.4.3)
> tests/continuous_linear_system/stability

A system with no null eigenvalues has a unique equilibrium state. (3.3, 3.4.6)
A system with null eigenvalues has a none or infinite equilibrium states. (3.3)
> tests/continuous_linear_system/equilibrium

An asymptotically stable system gain for a step input is the value of its transfer function at zero.
> tests/continuous_linear_system/static_gain

#### Discrete time

A linear system whose eigenvalues have modulus less than one is stable. (8.5.3)

A system with no eigenvalues with modulus equal to one has a unique equilibrium state. (8.4.7)

A system with eigenvalues with modulus equal to one has a none or infinite equilibrium states. (8.4.7)

An asymptotically stable transfer function gain for a step input is the value at one. (9.2.5)

A Finite Impulse Response (FIR) system output response to an impulse is zero after a number of steps equal to its order. (9.3.5)

A Finite Impulse Response (FIR) system output response to a step is equal to its gain after a number of steps equal to its order. (9.3.5)

## Plots

A Bode plot of a system with two conjugate complex poles shall have a resonance peak.
> tests/plots/bode_plot
