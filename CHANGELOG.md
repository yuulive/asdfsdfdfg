# Changelog

## Unreleased
### Added
- Discretization of transfer functions
### Changed
- Use typed unit of measure instead of primitive types

## [0.4.1] - 2019-09-01
### Added
- Documentation links for discrete system.
### Changed
- Applied clippy pedantic suggestions.

## [0.4.0] - 2019-08-26
### Added
- Implemented Runge-Kutta solver of order 4.
- Discrete linear systems time evolution.
- Discretization of continuous linear system.
- Allow to pass closures as input for the time evolution of a system.
- Example for system discretization.
### Changed
- Improve efficiency using LU decomposition to solve implicit system.
### Fixed
- Corrected the trasformation from transfer function to state-space form.

## [0.3.0] - 2019-08-05
### Added
- Radau implicit ordinary differential equations solver.
- Crate and module documentation.
- Example for stiff system.
### Fixed
- Calculation time inside ordinary differential equations solvers.

## [0.2.1] - 2019-08-01
### Changed
- Time evolution method requires a function as input.
- Add tolerance to the adaptive step size Runge-Kutta solver (rkf45) as parameter.
- Use time limit for the rkf45 solver.
### Fixed
- The output of the system is calculate with the time at the end of the step in the Runge-Kutta solvers.

## [0.2.0] - 2019-07-27
### Added
- First release with initial development
