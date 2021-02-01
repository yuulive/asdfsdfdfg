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
//! [Matrix of polynomials](polynomial_matrix/index.html)
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

#![warn(
    missing_crate_level_docs,
    missing_debug_implementations,
    missing_docs,
    unreachable_pub
)]

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(not(test))]
pub extern crate approx;
pub extern crate nalgebra;
pub extern crate num_complex;
pub extern crate num_traits;

pub mod complex;
pub mod controller;
pub mod enums;
pub mod error;
mod iterator;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod polynomial_matrix;
pub mod rational_function;
pub mod signals;
pub mod transfer_function;
pub mod units;

// Export from crate root.
pub use crate::complex::{damp, pulse};
pub use crate::enums::{Continuous, Discrete, Discretization, Time};
pub use crate::error::Error;
pub use crate::linear_system::{continuous::Ss, discrete::Ssd};
pub use crate::polynomial::Poly;
pub use crate::transfer_function::{
    continuous::Tf, discrete::Tfz, discretization::TfDiscretization, matrix::TfMatrix,
};
pub use crate::units::{Decibel, Hertz, RadiansPerSecond, Seconds};
