//! Polar plot
//!
//! Polar plot returns the iterator providing the complex numbers at the given
//! angular frequencies.
//!
//! Functions use angular frequencies as default inputs.

use crate::{transfer_function::Tf, units::RadiantsPerSecond, Eval};

use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd};

/// Struct for the calculation of Polar plots
#[derive(Debug)]
pub struct PolarIterator<T: Float> {
    /// Transfer function
    tf: Tf<T>,
    /// Number of intervals of the plot
    intervals: T,
    /// Step between frequencies
    step: T,
    /// Start frequency exponent
    base_freq_exp: T,
    /// Current data index
    index: T,
}

impl<T: Float + MulAdd<Output = T>> PolarIterator<T> {
    /// Create a PolarIterator struct
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
    /// is not lower than the maximum frequency
    pub(crate) fn new(
        tf: Tf<T>,
        min_freq: RadiantsPerSecond<T>,
        max_freq: RadiantsPerSecond<T>,
        step: T,
    ) -> Self {
        assert!(step > T::zero());
        assert!(min_freq < max_freq);

        let min = min_freq.0.log10();
        let max = max_freq.0.log10();
        let intervals = ((max - min) / step).floor();
        Self {
            tf,
            intervals,
            step,
            base_freq_exp: min,
            index: T::zero(),
        }
    }
}

/// Struct to hold the data returned by the Polar iterator
#[derive(Clone, Copy, Debug)]
pub struct Polar<T> {
    /// Output
    output: Complex<T>,
}

/// Implementation of Polar methods
impl<T: Float> Polar<T> {
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

/// Implementation of the Iterator trait for `PolarIterator` struct
impl<T: Float + MulAdd<Output = T>> Iterator for PolarIterator<T> {
    type Item = Polar<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = MulAdd::mul_add(self.step, self.index, self.base_freq_exp);
            // Casting is safe for both f32 and f64, representation is exact.
            let omega = T::from(10.0_f32).unwrap().powf(freq_exponent);
            let j_omega = Complex::<T>::new(T::zero(), omega);
            self.index = self.index + T::one();
            Some(Polar {
                output: self.tf.eval(&j_omega),
            })
        }
    }
}

/// Trait for the implementation of polar plot for a linear system.
pub trait PolarPlot<T: Float + FloatConst> {
    /// Create a PolarIterator struct
    ///
    /// # Arguments
    ///
    /// * `min_freq` - Minimum angular frequency of the plot
    /// * `max_freq` - Maximum angular frequency of the plot
    /// * `step` - Step between frequencies
    ///
    /// `step` shall be in logarithmic scale. Use 0.1 to have 10 point per decade
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum frequency
    /// is not lower than the maximum frequency
    fn polar(
        self,
        min_freq: RadiantsPerSecond<T>,
        max_freq: RadiantsPerSecond<T>,
        step: T,
    ) -> PolarIterator<T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly;

    #[test]
    fn create_iterator() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = PolarIterator::new(tf, RadiantsPerSecond(10.), RadiantsPerSecond(1000.), 0.1);
        assert_eq!(20., iter.intervals);
        assert_eq!(1., iter.base_freq_exp);
        assert_eq!(0., iter.index);
    }

    #[test]
    fn polar_struct() {
        let p = Polar {
            output: Complex::new(3., 4.),
        };
        assert_eq!(3., p.real());
        assert_eq!(4., p.imag());
        assert_eq!(5., p.magnitude());
        assert_relative_eq!(0.9273, p.phase(), max_relative = 0.00001);
    }

    #[test]
    fn iterator() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = PolarIterator::new(tf, RadiantsPerSecond(10.), RadiantsPerSecond(1000.), 0.1);
        // 20 steps -> 21 iteration
        assert_eq!(21, iter.count());
    }
}
