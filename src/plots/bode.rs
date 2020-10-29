//! # Bode plot
//!
//! Bode plot returns the angular frequency, the magnitude and the phase.
//!
//! Functions use angular frequencies as default inputs and output, being the
//! inverse of the poles and zeros time constants.

use num_traits::{Float, FloatConst, MulAdd, Num};

use crate::{
    plots::Plotter,
    units::{Hertz, RadiansPerSecond, ToDecibel},
};

/// Struct for the calculation of Bode plots
#[derive(Clone, Debug)]
pub struct Bode<T: Num, U: Plotter<T>> {
    /// Transfer function
    tf: U,
    /// Minimum angular frequency of the plot
    min_freq: RadiansPerSecond<T>,
    /// Maximum angular frequency of the plot
    max_freq: RadiansPerSecond<T>,
    /// Step between frequencies
    step: T,
}

impl<T: Float, U: Plotter<T>> Bode<T, U> {
    /// Create a `Bode` plot struct
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
    /// Panics if the step is not strictly positive and the minimum frequency
    /// is not lower than the maximum frequency
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

impl<T: Float + FloatConst, U: Plotter<T>> Bode<T, U> {
    /// Create a `Bode` plot struct for discrete time systems.
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
    /// Panics if the step is not strictly positive and the minimum frequency
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

impl<T: Float + MulAdd<Output = T>, U: Plotter<T>> IntoIterator for Bode<T, U> {
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
            base_freq: RadiansPerSecond(min),
            index: T::zero(),
        }
    }
}

/// Struct for the Polar plot data point iteration.
#[derive(Clone, Debug)]
pub struct IntoIter<T: Float, U: Plotter<T>> {
    /// Transfer function
    tf: U,
    /// Number of intervals of the plot
    intervals: T,
    /// Step between frequencies
    step: T,
    /// Start frequency
    base_freq: RadiansPerSecond<T>,
    /// Current data index
    index: T,
}

impl<T: Float + MulAdd<Output = T> + ToDecibel, U: Plotter<T>> IntoIter<T, U> {
    /// Convert `Bode` into decibels and degrees
    pub fn into_db_deg(self) -> impl Iterator<Item = Data<T>> {
        self.map(|g| Data {
            magnitude: g.magnitude.to_db(),
            phase: g.phase.to_degrees(),
            ..g
        })
    }
}

/// Struct to hold the data returned by the Bode iterator
#[derive(Debug, PartialEq)]
pub struct Data<T: Num> {
    /// Angular frequency (rad/s)
    angular_frequency: RadiansPerSecond<T>,
    /// Magnitude (absolute value or dB)
    magnitude: T,
    /// Phase (rad or degrees)
    phase: T,
}

impl<T: Float + FloatConst> Data<T> {
    /// Get the angular frequency
    pub fn angular_frequency(&self) -> RadiansPerSecond<T> {
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

/// Implementation of the Iterator trait for `Bode` struct
impl<T: Float + MulAdd<Output = T>, U: Plotter<T>> Iterator for IntoIter<T, U> {
    type Item = Data<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let freq_exponent = MulAdd::mul_add(self.step, self.index, self.base_freq.0);
            // Casting is safe for both f32 and f64, representation is exact.
            let omega = T::from(10.0_f32).unwrap().powf(freq_exponent);
            let g = self.tf.eval_point(omega);
            self.index = self.index + T::one();
            Some(Data {
                angular_frequency: RadiansPerSecond(omega),
                magnitude: g.norm(),
                phase: g.arg(),
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
        let iter = Bode::new(tf, RadiansPerSecond(10.), RadiansPerSecond(1000.), 0.1).into_iter();
        assert_relative_eq!(20., iter.intervals);
        assert_eq!(RadiansPerSecond(1.), iter.base_freq);
        assert_relative_eq!(0., iter.index);
    }

    #[test]
    fn create_iterator_db_deg() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = Bode::new(tf, RadiansPerSecond(10.), RadiansPerSecond(1000.), 0.1).into_iter();
        let iter2 = iter.into_db_deg();
        let res = iter2.last().unwrap();
        assert_eq!(RadiansPerSecond(1000.), res.angular_frequency());
        assert_relative_eq!(-90.0, res.phase(), max_relative = 0.001);
    }

    #[test]
    fn data_struct() {
        let f = RadiansPerSecond(120.);
        let mag = 3.;
        let ph = std::f64::consts::PI;
        let p = Data {
            angular_frequency: f,
            magnitude: mag,
            phase: ph,
        };
        assert_eq!(f, p.angular_frequency());
        assert_relative_eq!(19.0986, p.frequency().0, max_relative = 0.00001);
        assert_relative_eq!(mag, p.magnitude());
        assert_relative_eq!(ph, p.phase());
    }

    #[test]
    fn iterator() {
        let tf = Tf::new(poly!(2., 3.), poly!(1., 1., 1.));
        let iter = Bode::new(tf, RadiansPerSecond(10.), RadiansPerSecond(1000.), 0.1).into_iter();
        // 20 steps -> 21 iteration
        assert_eq!(21, iter.count());
    }
}
