//! # Bode plot
//!
//! Bode plot returns the angular frequency, the magnitude and the phase.
//!
//! Functions use angular frequencies as default inputs and output, being the
//! inverse of the poles and zeros time constants.

use crate::{
    transfer_function::Tf,
    units::{Decibel, Hertz, RadiantsPerSecond},
    Eval,
};

use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd};

/// Struct for the calculation of Bode plots
#[derive(Debug)]
pub struct BodeIterator<T: Float> {
    /// Transfer function
    tf: Tf<T>,
    /// Number of intervals of the plot
    intervals: T,
    /// Step between frequencies
    step: T,
    /// Start frequency
    base_freq: RadiantsPerSecond<T>,
    /// Current data index
    index: T,
}

impl<T: Decibel<T> + Float + MulAdd<Output = T>> BodeIterator<T> {
    /// Create a BodeIterator struct
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
            base_freq: RadiantsPerSecond(min),
            index: T::zero(),
        }
    }

    /// Convert BodeIterator into decibels and degrees
    pub fn into_db_deg(self) -> impl Iterator<Item = Bode<T>> {
        self.map(|g| Bode {
            magnitude: g.magnitude.to_db(),
            phase: g.phase.to_degrees(),
            ..g
        })
    }
}

/// Struct to hold the data returned by the Bode iterator
pub struct Bode<T: Float> {
    /// Angular frequency (rad)
    angular_frequency: RadiantsPerSecond<T>,
    /// Magnitude (absolute value or dB)
    magnitude: T,
    /// Phase (rad or degrees)
    phase: T,
}

/// Implementation of Bode methods
impl<T: Float + FloatConst> Bode<T> {
    /// Get the angular frequency
    pub fn angular_frequency(&self) -> RadiantsPerSecond<T> {
        self.angular_frequency
    }

    /// Get the frequency
    pub fn frequency(&self) -> Hertz<T> {
        self.angular_frequency.into()
    }

    /// Get the magnitude
    pub fn magnitude(&self) -> T {
        self.magnitude
    }

    /// Get the phase
    pub fn phase(&self) -> T {
        self.phase
    }
}

/// Implementation of the Iterator trait for `BodeIterator` struct
impl<T: Float + MulAdd<Output = T>> Iterator for BodeIterator<T> {
    type Item = Bode<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = MulAdd::mul_add(self.step, self.index, self.base_freq.0);
            // Casting is safe for both f32 and f64.
            let omega = T::from(10.).unwrap().powf(freq_exponent);
            let j_omega = Complex::<T>::new(T::zero(), omega);
            let g = self.tf.eval(&j_omega);
            //self.index += T::one();
            self.index = self.index + T::one();
            Some(Bode {
                angular_frequency: RadiantsPerSecond(omega),
                magnitude: g.norm(),
                phase: g.arg(),
            })
        }
    }
}

/// Trait for the implementation of Bode plot for a linear system.
pub trait BodePlot<T: Float + FloatConst> {
    /// Create a BodeIterator struct
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
    fn bode(
        self,
        min_freq: RadiantsPerSecond<T>,
        max_freq: RadiantsPerSecond<T>,
        step: T,
    ) -> BodeIterator<T>;

    /// Create a BodeIterator struct
    ///
    /// # Arguments
    ///
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
    fn bode_hz(self, min_freq: Hertz<T>, max_freq: Hertz<T>, step: T) -> BodeIterator<T>
    where
        Self: std::marker::Sized,
    {
        self.bode(min_freq.into(), max_freq.into(), step)
    }
}
