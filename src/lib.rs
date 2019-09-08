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

pub mod controller;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod transfer_function;

use std::convert::From;
const TAU: f64 = 2. * std::f64::consts::PI;

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

/// Trait for the conversion to decibels.
pub trait Decibel<T> {
    /// Convert to decibels
    fn to_db(&self) -> T;
}

/// Implementation of the Decibels for f64
impl Decibel<f64> for f64 {
    /// Convert f64 to decibels
    fn to_db(&self) -> Self {
        20. * self.log10()
    }
}

/// Unit of measure: seconds [s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Seconds(pub f64);

/// Unit of measure: Hertz [Hz] = [1/s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hertz(pub f64);

/// Unit of measure: Radiants per seconds [rad/s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RadiantsPerSecond(pub f64);

impl From<Hertz> for RadiantsPerSecond {
    fn from(hz: Hertz) -> Self {
        Self(TAU * hz.0)
    }
}

impl From<RadiantsPerSecond> for Hertz {
    fn from(rps: RadiantsPerSecond) -> Self {
        Self(rps.0 / TAU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ops::inv::Inv;

    #[test]
    fn decibel() {
        assert_eq!(40., 100_f64.to_db());
        assert_eq!(-3.0102999566398116, 2_f64.inv().sqrt().to_db());
    }

    #[test]
    fn conversion() {
        assert_eq!(RadiantsPerSecond(TAU), RadiantsPerSecond::from(Hertz(1.0)));

        let hz = Hertz(2.0);
        assert_eq!(hz, Hertz::from(RadiantsPerSecond::from(hz)));

        let rps = RadiantsPerSecond(2.0);
        assert_eq!(rps, RadiantsPerSecond::from(Hertz::from(rps)));
    }
}
