//! # Discretization module for continuous time transfer functions
//!
//! Given a continuous time transfer function, the sampling period and the
//! discretization method the `TfDiscretization` returns the evaluation of
//! the equivalent discrete time transfer function.
//!
//! The available discretization methods are forward Euler, backward Euler
//! and Tustin (Trapezoidal).

use num_complex::Complex;
use num_traits::{Float, Num};

use std::fmt::Debug;

use crate::{
    complex,
    enums::Discretization,
    polynomial::Poly,
    transfer_function::{continuous::Tf, discrete::Tfz},
    units::{RadiansPerSecond, Seconds},
};

/// Discretization of a transfer function
#[derive(Debug)]
pub struct TfDiscretization<T: Num> {
    /// Transfer function
    tf: Tf<T>,
    /// Sampling period
    ts: Seconds<T>,
    /// Discretization function
    conversion: fn(Complex<T>, Seconds<T>) -> Complex<T>,
}

/// Implementation of `TfDiscretization` struct.
impl<T: Num> TfDiscretization<T> {
    /// Create a new discretization transfer function from a continuous one.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `conversion` - Conversion function
    fn new_from_cont(
        tf: Tf<T>,
        ts: Seconds<T>,
        conversion: fn(Complex<T>, Seconds<T>) -> Complex<T>,
    ) -> Self {
        Self { tf, ts, conversion }
    }
}

/// Implementation of `TfDiscretization` struct.
impl<T: Float> TfDiscretization<T> {
    /// Discretize a transfer function.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `method` - Discretization method
    ///
    /// Example
    /// ```
    /// use automatica::{
    ///     polynomial::Poly,
    ///     transfer_function::discretization::TfDiscretization,
    ///     Discretization,
    ///     Seconds,
    ///     Tf
    /// };
    /// use automatica::num_complex::Complex64;
    /// let tf = Tf::new(
    ///     Poly::new_from_coeffs(&[2., 20.]),
    ///     Poly::new_from_coeffs(&[1., 0.1]),
    /// );
    /// let tfz = TfDiscretization::discretize(tf, Seconds(1.), Discretization::BackwardEuler);
    /// let gz = tfz.eval(Complex64::i());
    /// ```
    pub fn discretize(tf: Tf<T>, ts: Seconds<T>, method: Discretization) -> Self {
        let conv = match method {
            Discretization::ForwardEuler => fe,
            Discretization::BackwardEuler => fb,
            Discretization::Tustin => tu,
        };
        Self::new_from_cont(tf, ts, conv)
    }
}

impl<T: Float> Tf<T> {
    /// Convert a continuous time transfer function into a discrete time
    /// transfer function using the given method.
    ///
    /// * `ts` - Sampling period in seconds
    /// * `method` - Discretization method
    ///
    /// Example
    /// ```
    /// use automatica::{polynomial::Poly, Discretization, Seconds, Tf, Tfz};
    /// use automatica::num_complex::Complex64;
    /// let tf = Tf::new(
    ///     Poly::new_from_coeffs(&[2., 20.]),
    ///     Poly::new_from_coeffs(&[1., 0.1]),
    /// );
    /// let tfz = tf.discretize(Seconds(1.), Discretization::BackwardEuler);
    /// assert_eq!(0.1 / 1.1, tfz.real_poles().unwrap()[0]);
    /// ```
    pub fn discretize(&self, ts: Seconds<T>, method: Discretization) -> Tfz<T> {
        match method {
            Discretization::ForwardEuler => {
                let t = ts.0.recip();
                let s = Poly::new_from_coeffs(&[-t, t]);
                let num = self.num().eval_by_val(s.clone());
                let den = self.den().eval_by_val(s);
                Tfz::new(num, den)
            }
            Discretization::BackwardEuler => {
                let s_num = Poly::new_from_coeffs(&[-T::one(), T::one()]);
                let s_den = Poly::new_from_coeffs(&[T::zero(), ts.0]);
                discr_impl(self, &s_num, &s_den)
            }
            Discretization::Tustin => {
                let k = (T::one() + T::one()) / ts.0;
                let s_num = Poly::new_from_coeffs(&[-T::one(), T::one()]) * k;
                let s_den = Poly::new_from_coeffs(&[T::one(), T::one()]);
                discr_impl(self, &s_num, &s_den)
            }
        }
    }

    /// Convert a continuous time transfer function into a discrete time
    /// transfer function using Tustin method with frequency pre-warping.
    ///
    /// * `ts` - Sampling period in seconds
    /// * `warp_freq` - Pre-warping frequency in radians per second
    ///
    /// Example
    /// ```
    /// use automatica::{polynomial::Poly, Discretization, RadiansPerSecond, Seconds, Tf, Tfz};
    /// use automatica::num_complex::Complex64;
    /// let tf = Tf::new(
    ///     Poly::new_from_coeffs(&[2.0_f32, 20.]),
    ///     Poly::new_from_coeffs(&[1., 0.1]),
    /// );
    /// let tfz = tf.discretize_with_warp(Seconds(1.), RadiansPerSecond(0.1));
    /// assert_eq!(-0.6668982, tfz.real_poles().unwrap()[0]);
    /// ```
    pub fn discretize_with_warp(&self, ts: Seconds<T>, warp_freq: RadiansPerSecond<T>) -> Tfz<T> {
        let two = T::one() + T::one();
        let k = warp_freq.0 / (warp_freq.0 * ts.0 / two).tan();
        let s_num = Poly::new_from_coeffs(&[-T::one(), T::one()]) * k;
        let s_den = Poly::new_from_coeffs(&[T::one(), T::one()]);
        discr_impl(self, &s_num, &s_den)
    }
}

/// Common operations for discretization
#[allow(clippy::cast_sign_loss)]
fn discr_impl<T: Float>(tf: &Tf<T>, s_num: &Poly<T>, s_den: &Poly<T>) -> Tfz<T> {
    let s = Tf::new(s_num.clone(), s_den.clone());
    let num = tf.num().eval(&s).num().clone();
    let den = tf.den().eval(&s).num().clone();
    match tf.relative_degree() {
        g if g > 0 => {
            let num = num * s_den.powi(g as u32);
            Tfz::new(num, den)
        }
        g if g < 0 => {
            let den = den * s_num.powi(-g as u32);
            Tfz::new(num, den)
        }
        _ => Tfz::new(num, den),
    }
}

/// Forward Euler transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn fe<T: Float>(z: Complex<T>, ts: Seconds<T>) -> Complex<T> {
    (z - T::one()) / ts.0
}

/// Backward Euler transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn fb<T: Float>(z: Complex<T>, ts: Seconds<T>) -> Complex<T> {
    (z - T::one()) / (z * ts.0)
}

/// Tustin transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn tu<T: Float>(z: Complex<T>, ts: Seconds<T>) -> Complex<T> {
    let float = (T::one() + T::one()) / ts.0;
    let complex = complex::compdiv(z - T::one(), z + T::one());
    // Complex<T> * T is implemented, not T * Complex<T>
    complex * float
}

impl<T: Float> TfDiscretization<T> {
    /// Evaluate the discretization of the transfer function
    ///
    /// # Arguments
    /// * `z` - Value at which the transfer function is evaluated.
    pub fn eval(&self, z: Complex<T>) -> Complex<T> {
        let s = (self.conversion)(z, self.ts);
        self.tf.eval(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{polynomial::Poly, units::ToDecibel};
    use num_complex::Complex64;

    #[test]
    fn new_tfz() {
        let _t = TfDiscretization {
            tf: Tf::new(
                Poly::new_from_coeffs(&[0., 1.]),
                Poly::new_from_coeffs(&[1., 1., 1.]),
            ),
            ts: Seconds(0.1),
            conversion: |_z, _ts| 1.0 * Complex64::i(),
        };
    }

    #[test]
    fn eval_forward_euler() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let ts = Seconds(1.);
        let tfz = TfDiscretization::discretize(tf, ts, Discretization::ForwardEuler);
        let s = (tfz.conversion)(z, ts);
        assert_relative_eq!(-1.0, s.re);
        assert_relative_eq!(0.5, s.im);

        let gz = tfz.eval(z);
        assert_relative_eq!(27.175, gz.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(147.77, gz.arg().to_degrees(), max_relative = 1e-4);
    }

    #[test]
    fn eval_backward_euler() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let ts = Seconds(1.);
        let tfz = TfDiscretization::discretize(tf, ts, Discretization::BackwardEuler);
        let s = (tfz.conversion)(z, ts);
        assert_relative_eq!(1.0, s.re);
        assert_relative_eq!(2.0, s.im);

        let gz = tfz.eval(z);
        assert_relative_eq!(32.220, gz.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(50.884, gz.arg().to_degrees(), max_relative = 1e-4);
    }

    #[test]
    fn eval_tustin() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let ts = Seconds(1.);
        let tfz = TfDiscretization::discretize(tf, ts, Discretization::Tustin);
        let s = (tfz.conversion)(z, ts);
        assert_relative_eq!(-1.2, s.re);
        assert_relative_eq!(1.6, s.im);

        let gz = tfz.eval(z);
        assert_relative_eq!(32.753, gz.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(114.20, gz.arg().to_degrees(), max_relative = 1e-4);
    }

    #[test]
    fn discretization_forward_euler() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let tfz = tf
            .discretize(Seconds(1.), Discretization::ForwardEuler)
            .normalize();
        // (200z - 180)/(z + 9)
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[-180., 200.]),
            Poly::new_from_coeffs(&[9., 1.]),
        );
        assert_eq!(expected, tfz);
    }

    #[test]
    fn discretization_forward_euler2() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[1., 2., 3.]),
            Poly::new_from_coeffs(&[4., 5.]),
        );
        let tfz = tf
            .discretize(Seconds(0.1), Discretization::ForwardEuler)
            .normalize();
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[5.62, -11.6, 6.]),
            Poly::new_from_coeffs(&[-0.92, 1.]),
        );
        assert_eq!(expected, tfz);
    }

    #[test]
    fn discretization_forward_euler3() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[4.0, 5.]),
            Poly::new_from_coeffs(&[3., 2., 1.]),
        );
        let tfz = tf
            .discretize(Seconds(0.1), Discretization::ForwardEuler)
            .normalize();
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[-0.46, 0.5]),
            Poly::new_from_coeffs(&[0.83, -1.8, 1.]),
        );
        assert_eq!(expected, tfz);
    }

    #[test]
    fn discretization_backward_euler() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let tfz = tf
            .discretize(Seconds(1.), Discretization::BackwardEuler)
            .normalize();
        // (20z - 18.18)/(z - 0.09)
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[-20. / 1.1, 20.]),
            Poly::new_from_coeffs(&[-0.1 / 1.1, 1.]),
        );
        assert_eq!(expected, tfz);
    }

    #[test]
    fn discretization_tustin() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let tfz = tf
            .discretize(Seconds(1.), Discretization::Tustin)
            .normalize();
        // (35z - 31.67)/(z + 0.67)
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[-38. / 1.2, 35.]),
            Poly::new_from_coeffs(&[0.8 / 1.2, 1.]),
        );
        assert_eq!(expected, tfz);
    }

    #[test]
    fn frequency_warping() {
        // in scilab ss2tf(cls2dls(tf2ss(sys), 1, 0.1/2/%pi))
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2.0_f32, 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let tfz = tf
            .discretize_with_warp(Seconds(1.), RadiansPerSecond(0.1))
            .normalize();
        let expected = Tfz::new(
            Poly::new_from_coeffs(&[-31.643_282, 34.977_077]),
            Poly::new_from_coeffs(&[0.666_898_2, 1.]),
        );
        assert_eq!(expected, tfz);
    }
}
