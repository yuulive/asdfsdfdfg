use crate::{transfer_function::Tf, Decibel, Eval};
use num_complex::Complex64;

/// Struct for the calculation of Bode plots
#[derive(Debug)]
pub struct Bode {
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

impl Bode {
    /// Create a Bode struct
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to plot
    /// * `min_freq` - Minimum frequency of the plot
    /// * `max_freq` - Maximum frequency of the plot
    /// * `step` - Step between frequencies
    ///
    /// `step` shall be in logarithmic scale. Use 0.1 to have 10 point per decade
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum frequency
    /// is not lower than the maximum frequency
    pub fn new(tf: Tf, min_freq: f64, max_freq: f64, step: f64) -> Bode {
        assert!(step > 0.0);
        assert!(min_freq < max_freq);

        let min = min_freq.log10();
        let max = max_freq.log10();
        let intervals = ((max - min) / step).floor();
        Bode {
            tf,
            intervals,
            step,
            base_freq: min,
            index: 0.0,
        }
    }

    /// Convert Bode iterator into decibels and degrees
    pub fn into_db_deg(self) -> impl Iterator<Item = (f64, f64)> {
        self.map(|g| (g.0.to_db(), g.1.to_degrees()))
    }
}

/// Implementation of the Iterator trait for Bode struct
impl Iterator for Bode {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = self.step.mul_add(self.index, self.base_freq);
            let omega = Complex64::new(0.0, 10f64.powf(freq_exponent));
            let g = self.tf.eval(&omega);
            self.index += 1.;
            Some((g.norm(), g.arg()))
        }
    }
}
