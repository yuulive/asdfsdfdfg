---
title: Home page
version: 0.9.0
license: CC BY-SA 4.0
---

## Automatic Control Systems Library

This library allows the definition and the manipulation of time invariant linear systems (LTI) both continuous and discrete time.

The available representations for the LTI are:

- state representation: calculation of poles, equilibrium point, observability and controllability matrices, evolution of the system with time; for continuous time system the discretization is available;
- transfer function representation: calculation of poles and zeros, evaluation of the function, arithmetical operations, positive and negative feedback, static gain, initial value, sensitivity functions, discretization of continuous time transfer functions and ARMA representation for discrete time.

Conversions are available between the two representations.

This library contains also a module for the creation and manipulation of polynomials, necessary for the transfer functions, for which all the arithmetical operations, the derivative, the integration, the evaluation with real and complex numbers are defined, real and complex roots can be calculated.

The time evolution of continuous time systems is performed through different integrators, explicit, implicit and adaptive.

From the continuous time transfer functions it is possible to generate the data for Bode, polar and root locus plots.

It is also possible to define PID controllers.
