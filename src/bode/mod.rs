use crate::{transfer_function::Tf, Eval};
use num_complex::Complex64;

/// Struct for the calculation of Bode plots
#[derive(Debug)]
pub struct Bode {
    /// Transfer function
    tf: Tf,
    /// End of the plot
    stop: f64,
    /// Step between frequencies
    step: f64,
    /// Current frequency
    freq: f64,
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
    /// Panics if the step is not strictly positive
    pub fn new(tf: Tf, min_freq: f64, max_freq: f64, step: f64) -> Bode {
        assert!(step > 0.0);
        Bode {
            tf,
            stop: max_freq.log10(),
            step,
            freq: min_freq.log10(),
        }
    }
}

/// Implementation of the Iterator trait for Bode struct
impl Iterator for Bode {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.freq > self.stop {
            None
        } else {
            let omega = Complex64::new(0.0, 10f64.powf(self.freq));
            let g = self.tf.eval(&omega);
            self.freq += self.step;
            Some((g.norm(), g.arg()))
        }
    }
}
