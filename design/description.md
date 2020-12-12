---
title: Overall Description
version: 0.9.0
license: CC BY-SA 4.0
---

# Overall description

## Product perspective

### System interfaces

The library can be used on platforms on which the support for compilation exists.
For a reference to the available platforms it is possible to refer to the official list [14].

### User interfaces

This library is not designed to have a direct interaction with the user, therefore there are not user interfaces.

### Hardware Interfaces

This library does not directly interact with the hardware, it uses the function made available by the `Rust` standard library and the operating system in use.

### Software interfaces

The library needs the `rustc` compiler for the compilation with a version equal of greater than 1.44.

The list of the needed dependencies for the compilations of the library is the following:

- num-traits
    * 'trait' system for numeric types, allows the writing of generic numerical methods
    * source: <https://crates.io/crates/num-traits>
    * version: 0.2
- num-complex
    * complex numbers
    * source: <https://crates.io/crates/num-complex>
    * version: 0.3
- nalgebra
    * linear algebra, matrices, operations on matrices, decomposition and eigenvalues calculation
    * source: <https://crates.io/crates/nalgebra>
    * version: 0.23
- ndarray
    * multidimensional vectors, allows non numerical elements
    * source: <https://crates.io/crates/ndarray>
    * version: 0.14
- approx
    * approximate equality for floating pint numbers
    * source: <https://crates.io/crates/approx>
    * version: 0.4

The following dependencies are necessary for the development phase:

- quickcheck
    * randomized tests
    * source: <https://crates.io/crates/quickcheck>
    * version: 0.9
- quickcheck_macros
    * support macros for randomized tests
    * source: <https://crates.io/crates/quickcheck_macros>
    * version: 0.9

### Communications interfaces

This library does not have network interfaces.

### Memory

There are not memory limits for the use of this library.

## Product functions

The library allows the design by software of time invariant linear systems. It is possible to calculate is properties like the stability, the poles, the zeros and determine the time evolution of the system given an input and an initial state.

It is available the possibility to convert the representation of the system from states variables to transfer function and vice-versa.

Both discrete time and continuous time system are representable, continuous time systems can be discretized.

It is possible to generate the points for Bode diagrams, polar diagram and root locus diagram.

In the library a modulus for the polynomial calculation is available, include the arithmetical operations and the methods for the calculation of the roots of the polynomial.

## User characteristics

The user of this library is required to have engineering knowledge in the design of control systems.

## Constraints

There are non constraints in the use of this library.
Regarding the development, the correctness must be the first aim.
