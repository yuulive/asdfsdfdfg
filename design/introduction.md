# Introduction

## Purpose

This SRS (Software Requirements Specification) has the purpose of supplying the reference for the high level description of the software, of its design and its verification.

This document is addressed to those that intend to use this library in their software or those that intend to contribute to the development of the library.

## Scope

The name of the library is `automatica`. The chosen programming language for the development is `Rust` [11].

The version control system is `git` [12] and the source code is hosted on `Gitlab` at the following link:

[https://gitlab.com/daingun/automatica](https://gitlab.com/daingun/automatica)

The access to the public registry crates.io [13] is available at the following link:

[https://crates.io/crates/automatica](https://crates.io/crates/automatica)

The library supplies an infrastructure that contains calculation methods for the design of computer aided control systems. This software does not supply an interface with the final user.

The applications of this library should be low level (or system level) functions. The advantage is having fast and optimized low level calculations.

## Definitions, acronyms and abbreviations

*Polynomial*: algebraic sum of monomials made by a coefficient and a literal part;

*Time invariant linear system*: mathematical model of a physical object that interacts with the external world, whose functions of the state representation are linear and do not explicitly depend on time;

*State representation*: representation of a system through a state equation and an output transformation;

*Transfer function*: external representation of a system, it is equal to the ratio between the Laplace transforms of the forced output and the input that caused it;

*SISO*: single input single output, mono-variable system with only one input and output variable;

*MIMO*: multiple input multiple output, system with more than one input and output variables;

*by\_value*: the method parameter is passed as value;

*by\_ref*: the method parameter is passed as reference;

*in\ place*: the method parameter is passed as mutable reference;

*FR*: Functional requirement;

*TC*: Test case.

## Overview

This SRS is subdivided in a general description of the library that contains the interfaces and the requirements for the software, a description of specific and functional requirements of the library that describe what it is able to execute, a description of the test plan both black box and white box type, a description of the continuous integration process and the collection of the UML diagrams that describe the library structure.
