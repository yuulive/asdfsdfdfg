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

#[macro_use]
extern crate approx;

pub mod controller;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod transfer_function;
pub mod units;

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: &T) -> T;
}

// fn zipWith<U, C>(combo: C, left: U, right: U) -> impl Iterator
// where
//     U: Iterator,
//     C: FnMut(U::Item, U::Item) -> U::Item,
// {
//     left.zip(right).map(move |(l, r)| combo(l, r))
// }
/// Zip two slices with the given function
///
/// # Arguments
///
/// * `left` - first slice to zip
/// * `right` - second slice to zip
/// * `f` - function used to zip the two lists
pub(crate) fn zip_with<T, F>(left: &[T], right: &[T], mut f: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> T,
{
    left.iter().zip(right).map(|(l, r)| f(l, r)).collect()
}
