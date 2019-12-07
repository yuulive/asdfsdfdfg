//! Transfer functions for continuous time systems.

use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd};

use crate::{
    plots::{
        bode::{BodeIterator, BodePlot},
        polar::{PolarIterator, PolarPlot},
    },
    transfer_function::TfGen,
    units::{Decibel, RadiansPerSecond, Seconds},
    Continuous, Eval,
};

/// Continuous transfer function
pub type Tf<T> = TfGen<T, Continuous>;

impl<T: Float> Tf<T> {
    /// Time delay for continuous time transfer function.
    /// `y(t) = u(t - tau)`
    /// `G(s) = e^(-tau * s)
    ///
    /// # Arguments
    ///
    /// * `tau` - Time delay
    ///
    /// # Example
    /// ```
    /// use num_complex::Complex;
    /// use automatica::{units::Seconds, Tf};
    /// let d = Tf::delay(Seconds(2.));
    /// assert_eq!(1., d(Complex::new(0., 10.)).norm());
    /// ```
    pub fn delay(tau: Seconds<T>) -> impl Fn(Complex<T>) -> Complex<T> {
        move |s| (-s * tau.0).exp()
    }
}

impl<T: Float + MulAdd<Output = T>> Tf<T> {
    /// Static gain `G(0)`.
    /// Ratio between constant output and constant input.
    /// Static gain is defined only for transfer functions of 0 type.
    ///
    /// Example
    ///
    /// ```
    /// use automatica::{poly, Tf};
    /// let tf = Tf::new(poly!(4., -3.),poly!(2., 5., -0.5));
    /// assert_eq!(2., tf.static_gain());
    /// ```
    pub fn static_gain(&self) -> T {
        self.eval(&T::zero())
    }
}

/// Implementation of the Bode plot for a transfer function
impl<T: Decibel<T> + Float + FloatConst + MulAdd<Output = T>> BodePlot<T> for Tf<T> {
    fn bode(
        self,
        min_freq: RadiansPerSecond<T>,
        max_freq: RadiansPerSecond<T>,
        step: T,
    ) -> BodeIterator<T> {
        BodeIterator::new(self, min_freq, max_freq, step)
    }
}

/// Implementation of the polar plot for a transfer function
impl<T: Float + FloatConst + MulAdd<Output = T>> PolarPlot<T> for Tf<T> {
    fn polar(
        self,
        min_freq: RadiansPerSecond<T>,
        max_freq: RadiansPerSecond<T>,
        step: T,
    ) -> PolarIterator<T> {
        PolarIterator::new(self, min_freq, max_freq, step)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;

    use super::*;
    use crate::{poly, polynomial::Poly};

    #[test]
    fn delay() {
        let d = Tf::delay(Seconds(2.));
        assert_eq!(1., d(Complex::new(0., 10.)).norm());
        assert_eq!(-1., d(Complex::new(0., 0.5)).arg());
    }

    #[quickcheck]
    fn static_gain(g: f32) -> bool {
        let tf = Tf::new(poly!(g, -3.), poly!(1., 5., -0.5));
        g == tf.static_gain()
    }

    #[test]
    fn bode() {
        let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        let b = tf.bode(RadiansPerSecond(0.1), RadiansPerSecond(100.0), 0.1);
        for g in b.into_db_deg() {
            assert!(g.magnitude() < 0.);
            assert!(g.phase() < 0.);
        }
    }

    #[test]
    fn polar() {
        let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));
        let p = tf.polar(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
        for g in p {
            assert!(g.magnitude() < 1.);
            assert!(g.phase() < 0.);
        }
    }
}
