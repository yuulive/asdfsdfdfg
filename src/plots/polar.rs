//! Polar plot
//!
//! Polar plot returns the iterator providing the complex numbers at the given
//! angular frequencies.
//!
//! Functions use angular frequencies as default inputs.

use crate::{transfer_function::Tf, Eval};

use num_complex::Complex64;

/// Struct for the calculation of Polar plots
#[derive(Debug)]
pub struct PolarIterator {
    /// Transfer function
    tf: Tf,
    /// Number of intervals of the plot
    intervals: f64,
    /// Step between frequencies
    step: f64,
    /// Start frequency
    base_freq: f64,
    /// Current data index
    index: f64,
}

impl PolarIterator {
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
    pub(crate) fn new(tf: Tf, min_freq: f64, max_freq: f64, step: f64) -> Self {
        assert!(step > 0.0);
        assert!(min_freq < max_freq);

        let min = min_freq.log10();
        let max = max_freq.log10();
        let intervals = ((max - min) / step).floor();
        Self {
            tf,
            intervals,
            step,
            base_freq: min,
            index: 0.0,
        }
    }
}

/// Struct to hold the data returned by the Polar iterator
pub struct Polar {
    /// Output
    output: Complex64,
}

/// Implementation of Polar methods
impl Polar {
    /// Get the real part
    pub fn real(&self) -> f64 {
        self.output.re
    }

    /// Get the imaginary part
    pub fn imag(&self) -> f64 {
        self.output.im
    }

    /// Get the magnitude
    pub fn magnitude(&self) -> f64 {
        self.output.norm()
    }

    /// Get the phase
    pub fn phase(&self) -> f64 {
        self.output.arg()
    }
}

/// Implementation of the Iterator trait for `PolarIterator` struct
impl Iterator for PolarIterator {
    type Item = Polar;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = self.step.mul_add(self.index, self.base_freq);
            let omega = 10f64.powf(freq_exponent);
            let jomega = Complex64::new(0.0, omega);
            self.index += 1.;
            Some(Polar {
                output: self.tf.eval(&jomega),
            })
        }
    }
}

/// Trait for the implementation of polar plot for a linear system.
pub trait PolarPlot {
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
    fn polar(self, min_freq: f64, max_freq: f64, step: f64) -> PolarIterator;
}
