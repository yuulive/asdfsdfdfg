//! # Automatic Control Systems Library
//!
//! ## State-Space representation
//!
//! [Generic state space](linear_system/index.html)
//!
//! [Continuous](linear_system/continuous/index.html)
//!
//! [Discrete](linear_system/discrete/index.html)
//!
//! [Solvers](linear_system/solver/index.html)
//!
//! ## Transfer function representation
//!
//! [Generic transfer function](transfer_function/index.html)
//!
//! [Continuous](transfer_function/continuous/index.html)
//!
//! [Discrete](transfer_function/discrete/index.html)
//!
//! [Matrix of transfer functions](transfer_function/matrix/index.html)
//!
//! ## Plots
//!
//! [Bode plot](plots/bode/index.html)
//!
//! [Polar plot](plots/polar/index.html)
//!
//! [Root locus](plots/root_locus/index.html)
//!
//! ## Controllers
//!
//! [Pid](controller/pid/struct.Pid.html)
//!
//! ## Polynomials
//!
//! [Polynomials](polynomial/index.html)
//!
//! [Matrix of polynomials](polynomial/matrix/index.html)
//!
//! ## Units of measurement
//!
//! [Units](units/index.html)
//!
//! ## Signals
//!
//! [Continuous](signals/continuous/index.html)
//!
//! [Discrete](signals/discrete/index.html)

#![warn(missing_docs)]

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod controller;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod signals;
pub mod transfer_function;
pub mod units;
pub mod utils;

pub use linear_system::{continuous::Ss, discrete::Ssd};
pub use polynomial::Poly;
pub use transfer_function::{
    continuous::Tf, discrete::Tfz, discretization::TfDiscretization, matrix::TfMatrix,
};

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: T) -> T {
        Eval::eval_ref(self, &x)
    }

    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval_ref(&self, x: &T) -> T;
}

/// Trait to tag Continuous or Discrete types
pub trait Time {}

/// Type for continuous systems
#[derive(Debug, PartialEq)]
pub enum Continuous {}
impl Time for Continuous {}

/// Type for discrete systems
#[derive(Debug, PartialEq)]
pub enum Discrete {}
impl Time for Discrete {}
