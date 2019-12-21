//! # Transfer function discretization
//!
//! The discretization can be performed with Euler or Tustin methods.

use crate::{
    linear_system::discrete::Discretization,
    transfer_function::{continuous::Tf, TfGen},
    units::Seconds,
    Discrete, Eval,
};

use num_complex::Complex;
use num_traits::{Float, MulAdd, Num};

/// Discrete transfer function
pub type Tfz<T> = TfGen<T, Discrete>;

impl<T: Float> Tfz<T> {
    /// Time delay for discrete time transfer function.
    /// `y(k) = u(k - h)`
    /// `G(z) = z^(-h)
    ///
    /// # Arguments
    ///
    /// * `h` - Time delay
    ///
    /// # Example
    /// ```
    /// use num_complex::Complex;
    /// use automatica::{units::Seconds, Tfz};
    /// let d = Tfz::delay(2);
    /// assert_eq!(0.010000001, d(Complex::new(0., 10.0_f32)).norm());
    /// ```
    pub fn delay(k: i32) -> impl Fn(Complex<T>) -> Complex<T> {
        move |z| z.powi(-k)
    }

    /// System inital value response to step input.
    /// `y(0) = G(z->infinity)`
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tf = Tfz::new(poly!(4.), poly!(1., 5.));
    /// assert_eq!(0., tf.init_value());
    /// ```
    pub fn init_value(&self) -> T {
        let n = self.num.degree();
        let d = self.den.degree();
        if n < d {
            T::zero()
        } else if n == d {
            self.num.leading_coeff() / self.den.leading_coeff()
        } else {
            T::infinity()
        }
    }
}

impl<T: Float + MulAdd<Output = T>> Tfz<T> {
    /// Static gain `G(1)`.
    /// Ratio between constant output and constant input.
    /// Static gain is defined only for transfer functions of 0 type.
    ///
    /// Example
    ///
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tf = Tfz::new(poly!(5., -3.),poly!(2., 5., -6.));
    /// assert_eq!(2., tf.static_gain());
    /// ```
    pub fn static_gain(&self) -> T {
        self.eval(&T::one())
    }
}

use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::Mul;
//impl<T: Float + Mul<Output = T> + Sum> Tfz<T> {
impl<T: Debug + Float + Mul<Output = T> + Sum> Tfz<T> {
    //pub fn arma(&self) -> impl Fn(T) -> T {
    pub fn arma(&self) -> impl FnMut(T) -> T {
        let mut g = self.normalize();
        let n = g.den.degree().unwrap_or(1);

        // The front is the lowest order coefficient.
        // The back is the higher order coefficient.
        // The last coefficient is always 1.
        // [a0, a1, a2, ..., a(n-1), 1]
        let y_coeffs = g.den.coeffs;
        // [b0, b1, b2, ..., bn]
        // The numerator must be extended to the degree of the denominator
        // and the higher degree terms (more recent) must be zero.
        g.num.extend(n);
        let u_coeffs = g.num.coeffs;

        // The front is the oldest calculated output.
        // [y(k-n), y(k-n+1), ..., y(k-1), y(k)]
        let mut y = VecDeque::from(vec![T::zero(); y_coeffs.len()]);
        // The front is the oldest input.
        // [u(k-n), u(k-n+1), ..., u(k-1), u(k)]
        let mut u = VecDeque::from(vec![T::zero(); u_coeffs.len()]);
        debug_assert!(u_coeffs.len() == u.len());
        debug_assert!(u.len() == y.len());

        move |uk| {
            u.push_back(uk);
            u.pop_front();
            let input: T = u_coeffs.iter().zip(&u).map(|(&i, &j)| i * j).sum();

            y.push_back(T::zero());
            y.pop_front();
            let old_output: T = y_coeffs.iter().zip(&y).map(|(&i, &j)| i * j).sum();

            let new_y = input - old_output;
            if let Some(x) = y.back_mut() {
                *x = new_y;
            }
            new_y
        }
    }
}

/// Discretization of a transfer function
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
    ///     linear_system::discrete::Discretization,
    ///     polynomial::Poly,
    ///     transfer_function::discrete::TfDiscretization,
    ///     units::Seconds,
    ///     Eval,
    ///     Tf
    /// };
    /// use num_complex::Complex64;
    /// let tf = Tf::new(
    ///     Poly::new_from_coeffs(&[2., 20.]),
    ///     Poly::new_from_coeffs(&[1., 0.1]),
    /// );
    /// let tfz = TfDiscretization::discretize(tf, Seconds(1.), Discretization::BackwardEuler);
    /// let gz = tfz.eval(&Complex64::i());
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
    let complex = (z - T::one()) / (z + T::one());
    // Complex<T> * T is implemented, not T * Complex<T>
    complex * float
}

/// Implementation of the evaluation of a transfer function discretization
impl<T: Float + MulAdd<Output = T>> Eval<Complex<T>> for TfDiscretization<T> {
    fn eval(&self, z: &Complex<T>) -> Complex<T> {
        let s = (self.conversion)(*z, self.ts);
        self.tf.eval(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, polynomial::Poly, units::Decibel, Eval};
    use num_complex::Complex64;

    #[test]
    fn tfz() {
        let _ = Tfz::new(poly!(1.), poly!(1., 2., 3.));
    }

    #[test]
    fn delay() {
        let d = Tfz::delay(2);
        assert_eq!(0.010_000_001, d(Complex::new(0., 10.0_f32)).norm());
    }

    #[test]
    fn initial_value() {
        let tf = Tfz::new(poly!(4.), poly!(1., 5.));
        assert_eq!(0., tf.init_value());

        let tf = Tfz::new(poly!(4., 10.), poly!(1., 5.));
        assert_eq!(2., tf.init_value());

        let tf = Tfz::new(poly!(4., 1.), poly!(5.));
        assert_eq!(std::f32::INFINITY, tf.init_value());
    }

    #[test]
    fn static_gain() {
        let tf = Tfz::new(poly!(5., -3.), poly!(2., 5., -6.));
        assert_eq!(2., tf.static_gain());
    }

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
    fn eval() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let g = tf.eval(&z);
        assert_relative_eq!(20.159, g.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(75.828, g.arg().to_degrees(), max_relative = 1e-4);
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
        assert_eq!(Complex64::new(-1.0, 0.5), s);

        let gz = tfz.eval(&z);
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
        assert_eq!(Complex64::new(1.0, 2.0), s);

        let gz = tfz.eval(&z);
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
        assert_eq!(Complex64::new(-1.2, 1.6), s);

        let gz = tfz.eval(&z);
        assert_relative_eq!(32.753, gz.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(114.20, gz.arg().to_degrees(), max_relative = 1e-4);
    }

    #[test]
    fn arma() {
        let tfz = Tfz::new(poly!(0.5_f32), poly!(-0.5, 1.));
        let mut sys = tfz.arma();

        assert_eq!(0.0, sys(1.));
        assert_eq!(0.5, sys(0.));
        assert_eq!(0.25, sys(0.));
        assert_eq!(0.125, sys(0.));
        assert_eq!(0.0625, sys(0.));
        assert_eq!(0.03125, sys(0.));
    }
}
