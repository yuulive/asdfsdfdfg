//! # Frequency response plots
//!
//! [Bode plot](bode/index.html)
//!
//! [Polar plot](polar/index.html)
//!
//! [Root locus](root_locus/index.html)
//!
//! Plots are implemented as iterators.

pub mod bode;
pub mod polar;
pub mod root_locus;

use num_complex::Complex;

/// Determine how the transfer function is evaluated in plots.
pub trait Plotter<T> {
    /// Evaluate the transfer function at the given value.
    ///
    /// # Arguments
    ///
    /// * `x` - value at which the function is evaluated
    fn eval_point(&self, x: T) -> Complex<T>;
}
