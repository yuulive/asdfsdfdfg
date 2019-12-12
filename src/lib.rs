//! # Automatic Control Systems Library
//!
//! ## State-Space representation
//!
//! [State space representation](linear_system/struct.Ss.html)
//!
//! [Equilibrium](linear_system/struct.Equilibrium.html)
//!
//! ### System time evolution
//!
//! [Solvers](linear_system/solver/index.html)
//!
//! ## Discrete system
//!
//! [Discrete](linear_system/discrete/index.html)
//!
//! [Transfer function discretization](transfer_function/discrete_tf/index.html)
//!
//! ## Transfer function representation
//!
//! [Transfer function](transfer_function/struct.Tf.html)
//!
//! [Matrix of transfer functions](transfer_function/struct.TfMatrix.html)
//!
//! ## Plots
//!
//! [Bode plot](plots/bode/index.html)
//!
//! [Polar plot](plots/polar/index.html)
//!
//! ## Controllers
//!
//! [Pid](controller/pid/struct.Pid.html)
//!
//! ## Polynomials
//!
//! [Polynomials](polynomial/struct.Poly.html)
//!
//! [Matrix of polynomials](polynomial/struct.MatrixOfPoly.html)

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
pub use transfer_function::{continuous::Tf, discrete::Tfz, matrix::TfMatrix};

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: &T) -> T;
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
