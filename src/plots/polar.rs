//! # Polar plot
//!
//! Polar plot returns the iterator providing the complex numbers at the given
//! angular frequencies.
//!
//! Functions use angular frequencies as default inputs.

use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd, Num};

use crate::{plots::Plotter, units::RadiansPerSecond};

/// Struct representing a Polar plot.
#[derive(Clone, Debug)]
pub struct Polar<T: Num, U: Plotter<T>> {
    /// Transfer function
    tf: U,
    /// Minimum angular frequency of the plot
    min_freq: RadiansPerSecond<T>,
    /// Maximum angular frequency of the plot
    max_freq: RadiansPerSecond<T>,
    /// Step between frequencies
    step: T,
}

impl<T: Float + MulAdd<Output = T>, U: Plotter<T>> Polar<T, U> {
    /// Create a `Polar` plot struct
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to plot
    /// * `min_freq` - Minimum angular frequency of the plot
    /// * `max_freq` - Maximum angular frequency of the plot
    /// * `step` - Step between frequencies
    ///
    /// `step` shall be in logarithmic scale. Use 0.1 to have 10 point per decade
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum frequency
    /// is not lower than the maximum frequency.
    pub fn new(
        tf: U,
        min_freq: RadiansPerSecond<T>,
        max_freq: RadiansPerSecond<T>,
        step: T,
    ) -> Self {
        assert!(step > T::zero());
        assert!(min_freq < max_freq);

        Self {
            tf,
            min_freq,
            max_freq,
            step,
        }
    }
}

impl<T: Float + FloatConst + MulAdd<Output = T>, U: Plotter<T>> Polar<T, U> {
    /// Create a `Polar` plot struct
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to plot
    /// * `min_freq` - Minimum angular frequency of the plot
    /// * `step` - Step between frequencies
    ///
    /// `step` shall be in logarithmic scale. Use 0.1 to have 10 point per decade
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum frequency
    /// is not lower than pi.
    pub fn new_discrete(tf: U, min_freq: RadiansPerSecond<T>, step: T) -> Self {
        let pi = RadiansPerSecond(T::PI());
        assert!(step > T::zero());
        assert!(min_freq < pi);

        Self {
            tf,
            min_freq,
            max_freq: pi,
            step,
        }
    }
}

impl<T: Float + MulAdd<Output = T>, U: Plotter<T>> IntoIterator for Polar<T, U> {
    type Item = Data<T>;
    type IntoIter = IntoIter<T, U>;

    fn into_iter(self) -> Self::IntoIter {
        let min = self.min_freq.0.log10();
        let max = self.max_freq.0.log10();
        let intervals = ((max - min) / self.step).floor();
        Self::IntoIter {
            tf: self.tf,
            intervals,
            step: self.step,
            base_freq_exp: min,
            index: T::zero(),
        }
    }
}

/// Struct for the Polar plot data point iteration.
#[derive(Clone, Debug)]
pub struct IntoIter<T: Float + MulAdd<Output = T>, U: Plotter<T>> {
    /// Transfer function
    tf: U,
    /// Number of intervals of the plot
    intervals: T,
    /// Step between frequencies
    step: T,
    /// Start frequency exponent
    base_freq_exp: T,
    /// Current data index
    index: T,
}

/// Struct to hold the data returned by the Polar iterator.
#[derive(Clone, Copy, Debug)]
pub struct Data<T> {
    /// Output
    output: Complex<T>,
}

impl<T: Float> Data<T> {
    /// Get the output
    pub fn output(&self) -> Complex<T> {
        self.output
    }

    /// Get the real part
    pub fn real(&self) -> T {
        self.output.re
    }

    /// Get the imaginary part
    pub fn imag(&self) -> T {
        self.output.im
    }

    /// Get the magnitude
    pub fn magnitude(&self) -> T {
        self.output.norm()
    }

    /// Get the phase
    pub fn phase(&self) -> T {
        self.output.arg()
    }
}

/// Implementation of the Iterator trait for `Polar` struct
// impl<T: Float + MulAdd<Output = T>> Iterator for IntoIter<T, Continuous> {
impl<T: Float + MulAdd<Output = T>, U: Plotter<T>> Iterator for IntoIter<T, U> {
    type Item = Data<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = MulAdd::mul_add(self.step, self.index, self.base_freq_exp);
            // Casting is safe for both f32 and f64, representation is exact.
            let omega = T::from(10.0_f32).unwrap().powf(freq_exponent);
            self.index = self.index + T::one();
            Some(Data {
                output: self.tf.eval_point(omega),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, transfer_function::continuous::Tf};

    #[test]
    fn create_iterator() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = Polar::new(tf, RadiansPerSecond(10.), RadiansPerSecond(1000.), 0.1).into_iter();
        assert_relative_eq!(20., iter.intervals);
        assert_relative_eq!(1., iter.base_freq_exp);
        assert_relative_eq!(0., iter.index);
    }

    #[test]
    fn data_struct() {
        let c = Complex::new(3., 4.);
        let p = Data { output: c };
        assert_eq!(c, p.output());
        assert_relative_eq!(3., p.real());
        assert_relative_eq!(4., p.imag());
        assert_relative_eq!(5., p.magnitude());
        assert_relative_eq!(0.9273, p.phase(), max_relative = 0.00001);
    }

    #[test]
    fn iterator() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = Polar::new(tf, RadiansPerSecond(10.), RadiansPerSecond(1000.), 0.1).into_iter();
        // 20 steps -> 21 iteration
        assert_eq!(21, iter.count());
    }
}
