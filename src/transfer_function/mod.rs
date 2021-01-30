//! # Transfer function and matrices of transfer functions
//!
//! This module contains the generic methods for transfer functions
//! * calculation of zeros and poles (real and complex)
//! * arithmetic operations (addition, subtraction, multiplication, division,
//!   negation, inversion)
//! * positive and negative feedback
//! * conversion from a generic state-space representation of a single input
//!   single output system
//! * evaluation of the transfer function at the given complex number
//!
//! [continuous](continuous/index.html) module contains the specialized
//! structs and methods for continuous systems.
//!
//! [discrete](discrete/index.html) module contains the specialized structs and
//! methods for discrete systems.
//!

pub mod continuous;
pub mod discrete;
pub mod discretization;
pub mod matrix;

use nalgebra::RealField;
use num_complex::Complex;
use num_traits::{Float, Inv, One, Signed, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    enums::Time,
    error::{Error, ErrorKind},
    linear_system::{self, SsGen},
    polynomial::Poly,
    polynomial_matrix::{MatrixOfPoly, PolyMatrix},
};

/// Transfer function representation of a linear system
#[derive(Clone, Debug, PartialEq)]
pub struct TfGen<T, U: Time> {
    /// Transfer function numerator
    num: Poly<T>,
    /// Transfer function denominator
    den: Poly<T>,
    /// Tag to disambiguate continuous and discrete
    time: PhantomData<U>,
}

/// Implementation of transfer function methods
impl<T: Float, U: Time> TfGen<T, U> {
    /// Create a new transfer function given its numerator and denominator
    ///
    /// # Arguments
    ///
    /// * `num` - Transfer function numerator
    /// * `den` - Transfer function denominator
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tfz = Tfz::new(poly!(1., 2.), poly!(-4., 6., -2.));
    /// ```
    #[must_use]
    pub fn new(num: Poly<T>, den: Poly<T>) -> Self {
        Self {
            num,
            den,
            time: PhantomData::<U>,
        }
    }

    /// Extract transfer function numerator
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let num = poly!(1., 2.);
    /// let tfz = Tfz::new(num.clone(), poly!(-4., 6., -2.));
    /// assert_eq!(&num, tfz.num());
    /// ```
    #[must_use]
    pub fn num(&self) -> &Poly<T> {
        &self.num
    }

    /// Extract transfer function denominator
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let den = poly!(-4., 6., -2.);
    /// let tfz = Tfz::new(poly!(1., 2.), den.clone());
    /// assert_eq!(&den, tfz.den());
    /// ```
    #[must_use]
    pub fn den(&self) -> &Poly<T> {
        &self.den
    }

    /// Calculate the relative degree between denominator and numerator.
    ///
    /// # Example
    /// ```
    /// use automatica::{num_traits::Inv, poly, Tfz};
    /// let tfz = Tfz::new(poly!(1., 2.), poly!(-4., 6., -2.));
    /// let expected = tfz.relative_degree();
    /// assert_eq!(expected, 1);
    /// assert_eq!(tfz.inv().relative_degree(), -1);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn relative_degree(&self) -> i32 {
        match (self.den.degree(), self.num.degree()) {
            (Some(d), Some(n)) => d as i32 - n as i32,
            (Some(d), None) => d as i32,
            (None, Some(n)) => -(n as i32),
            _ => 0,
        }
    }
}

/// Implementation of transfer function methods
impl<T, U: Time> TfGen<T, U> {
    /// Compute the reciprocal of a transfer function in place.
    pub fn inv_mut(&mut self) {
        std::mem::swap(&mut self.num, &mut self.den);
    }
}

impl<T: Clone, U: Time> Inv for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    /// Compute the reciprocal of a transfer function.
    fn inv(self) -> Self::Output {
        Self::Output {
            num: self.den.clone(),
            den: self.num.clone(),
            time: PhantomData::<U>,
        }
    }
}

impl<T: Clone, U: Time> Inv for TfGen<T, U> {
    type Output = Self;

    /// Compute the reciprocal of a transfer function.
    fn inv(mut self) -> Self::Output {
        self.inv_mut();
        self
    }
}

impl<T: Float + RealField, U: Time> TfGen<T, U> {
    /// Calculate the poles of the transfer function
    #[must_use]
    pub fn real_poles(&self) -> Option<Vec<T>> {
        self.den.real_roots()
    }

    /// Calculate the poles of the transfer function
    #[must_use]
    pub fn complex_poles(&self) -> Vec<Complex<T>> {
        self.den.complex_roots()
    }

    /// Calculate the zeros of the transfer function
    #[must_use]
    pub fn real_zeros(&self) -> Option<Vec<T>> {
        self.num.real_roots()
    }

    /// Calculate the zeros of the transfer function
    #[must_use]
    pub fn complex_zeros(&self) -> Vec<Complex<T>> {
        self.num.complex_roots()
    }
}

impl<T: Float, U: Time> TfGen<T, U> {
    /// Negative feedback.
    ///
    /// ```text
    ///           L(s)
    /// G(s) = ----------
    ///         1 + L(s)
    /// ```
    /// where `self = L(s)`
    #[must_use]
    pub fn feedback_n(&self) -> Self {
        Self {
            num: self.num.clone(),
            den: &self.den + &self.num,
            time: PhantomData::<U>,
        }
    }

    /// Positive feedback
    ///
    /// ```text
    ///           L(s)
    /// G(s) = ----------
    ///         1 - L(s)
    /// ```
    /// where `self = L(s)`
    #[must_use]
    pub fn feedback_p(&self) -> Self {
        Self {
            num: self.num.clone(),
            den: &self.den - &self.num,
            time: PhantomData::<U>,
        }
    }

    /// Normalization of transfer function. If the denominator is zero the same
    /// transfer function is returned.
    ///
    /// from:
    /// ```text
    ///        b_n*z^n + b_(n-1)*z^(n-1) + ... + b_1*z + b_0
    /// G(z) = ---------------------------------------------
    ///        a_n*z^n + a_(n-1)*z^(n-1) + ... + a_1*z + a_0
    /// ```
    /// to:
    /// ```text
    ///        b'_n*z^n + b'_(n-1)*z^(n-1) + ... + b'_1*z + b'_0
    /// G(z) = -------------------------------------------------
    ///          z^n + a'_(n-1)*z^(n-1) + ... + a'_1*z + a'_0
    /// ```
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tfz = Tfz::new(poly!(1., 2.), poly!(-4., 6., -2.));
    /// let expected = Tfz::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
    /// assert_eq!(expected, tfz.normalize());
    /// ```
    #[must_use]
    pub fn normalize(&self) -> Self {
        if self.den.is_zero() {
            return self.clone();
        }
        let (den, an) = self.den.monic();
        let num = &self.num / an;
        Self {
            num,
            den,
            time: PhantomData,
        }
    }

    /// In place normalization of transfer function. If the denominator is zero
    /// no operation is done.
    ///
    /// from:
    /// ```text
    ///        b_n*z^n + b_(n-1)*z^(n-1) + ... + b_1*z + b_0
    /// G(z) = ---------------------------------------------
    ///        a_n*z^n + a_(n-1)*z^(n-1) + ... + a_1*z + a_0
    /// ```
    /// to:
    /// ```text
    ///        b'_n*z^n + b'_(n-1)*z^(n-1) + ... + b'_1*z + b'_0
    /// G(z) = -------------------------------------------------
    ///          z^n + a'_(n-1)*z^(n-1) + ... + a'_1*z + a'_0
    /// ```
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let mut tfz = Tfz::new(poly!(1., 2.), poly!(-4., 6., -2.));
    /// tfz.normalize_mut();
    /// let expected = Tfz::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
    /// assert_eq!(expected, tfz);
    /// ```
    pub fn normalize_mut(&mut self) {
        if self.den.is_zero() {
            return;
        }
        let an = self.den.monic_mut();
        self.num.div_mut(&an);
    }
}

macro_rules! from_ss_to_tr {
    ($ty:ty, $laverrier:expr) => {
        impl<U: Time> TfGen<$ty, U> {
            /// Convert a state-space representation into transfer functions.
            /// Conversion is available for Single Input Single Output system.
            /// If fails if the system is not SISO
            ///
            /// # Arguments
            ///
            /// `ss` - state space linear system
            ///
            /// # Errors
            ///
            /// It returns an error if the linear system is not single input
            /// single output.
            pub fn new_from_siso(ss: &SsGen<$ty, U>) -> Result<Self, Error> {
                let (pc, a_inv) = $laverrier(&ss.a);
                let g = a_inv.left_mul(&ss.c).right_mul(&ss.b);
                let rest = PolyMatrix::multiply(&pc, &ss.d);
                let tf = g + rest;
                if let Some(num) = MatrixOfPoly::from(tf).single() {
                    Ok(Self::new(num.clone(), pc))
                } else {
                    Err(Error::new_internal(ErrorKind::NoSisoSystem))
                }
            }
        }
    };
}

from_ss_to_tr!(f64, linear_system::leverrier_f64);
from_ss_to_tr!(f32, linear_system::leverrier_f32);

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Float, U: Time> Neg for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn neg(self) -> Self::Output {
        Self::Output {
            num: -&self.num,
            den: self.den.clone(),
            time: PhantomData,
        }
    }
}

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Float, U: Time> Neg for TfGen<T, U> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.num = -self.num;
        self
    }
}

/// Implementation of transfer function addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Add for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let (num, den) = if self.den == rhs.den {
            (&self.num + &rhs.num, self.den.clone())
        } else {
            (
                &self.num * &rhs.den + &self.den * &rhs.num,
                &self.den * &rhs.den,
            )
        };
        Self::Output::new(num, den)
    }
}

/// Implementation of transfer function addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Add for TfGen<T, U> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        if self.den == rhs.den {
            self.num = self.num + rhs.num;
            self
        } else {
            // first modify numerator.
            self.num = &self.num * &rhs.den + &self.den * &rhs.num;
            self.den = self.den * rhs.den;
            self
        }
    }
}

/// Implementation of transfer function addition
impl<T: Float, U: Time> Add<T> for TfGen<T, U> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of transfer function addition
impl<T: Float, U: Time> Add<&T> for TfGen<T, U> {
    type Output = Self;

    fn add(mut self, rhs: &T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of transfer function subtraction
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Sub for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return -rhs.clone();
        }
        if rhs.is_zero() {
            return self.clone();
        }
        let (num, den) = if self.den == rhs.den {
            (&self.num - &rhs.num, self.den.clone())
        } else {
            (
                &self.num * &rhs.den - &self.den * &rhs.num,
                &self.den * &rhs.den,
            )
        };
        Self::Output::new(num, den)
    }
}

/// Implementation of transfer function subtraction
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Sub for TfGen<T, U> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        if self.is_zero() {
            return -rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        if self.den == rhs.den {
            self.num = self.num - rhs.num;
            self
        } else {
            // first modify numerator.
            self.num = &self.num * &rhs.den - &self.den * &rhs.num;
            self.den = self.den * rhs.den;
            self
        }
    }
}

/// Implementation of transfer function multiplication
impl<T: Float, U: Time> Mul for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::Output::zero();
        }
        let num = &self.num * &rhs.num;
        let den = &self.den * &rhs.den;
        Self::Output::new(num, den)
    }
}

/// Implementation of transfer function multiplication
impl<T: Float, U: Time> Mul for TfGen<T, U> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::Output::zero();
        }
        self.num = self.num * rhs.num;
        self.den = self.den * rhs.den;
        self
    }
}

/// Implementation of transfer function multiplication
impl<T: Float, U: Time> Mul<&TfGen<T, U>> for TfGen<T, U> {
    type Output = Self;

    fn mul(mut self, rhs: &TfGen<T, U>) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::Output::zero();
        }
        self.num = self.num * &rhs.num;
        self.den = self.den * &rhs.den;
        self
    }
}

/// Implementation of transfer function division
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Div for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self.is_zero(), rhs.is_zero()) {
            (true, false) => return Self::Output::zero(),
            (false, true) => return Self::Output::zero().inv(),
            _ => (),
        };
        let num = &self.num * &rhs.den;
        let den = &self.den * &rhs.num;
        Self::Output::new(num, den)
    }
}

/// Implementation of transfer function division
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Div for TfGen<T, U> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self {
        match (self.is_zero(), rhs.is_zero()) {
            (true, false) => return Self::Output::zero(),
            (false, true) => return Self::Output::zero().inv(),
            _ => (),
        };
        self.num = self.num * rhs.den;
        self.den = self.den * rhs.num;
        self
    }
}

impl<T: Float, U: Time> Zero for TfGen<T, U> {
    fn zero() -> Self {
        Self {
            num: Poly::zero(),
            den: Poly::one(),
            time: PhantomData::<U>,
        }
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero() && !self.den.is_zero()
    }
}

impl<T: Clone, U: Time> TfGen<T, U> {
    /// Evaluate the transfer function.
    ///
    /// # Arguments
    ///
    /// * `s` - Value at which the transfer function is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tf};
    /// use automatica::num_complex::Complex as C;
    /// let tf = Tf::new(poly!(1., 2., 3.), poly!(-4., -3., 1.));
    /// assert_eq!(-8.5, tf.eval_by_val(3.));
    /// assert_eq!(C::new(0.64, -0.98), tf.eval_by_val(C::new(0., 2.0_f32)));
    /// ```
    pub fn eval_by_val<N>(&self, s: N) -> N
    where
        N: Add<T, Output = N> + Clone + Div<Output = N> + Mul<Output = N> + Zero,
    {
        self.num.eval_by_val(s.clone()) / self.den.eval_by_val(s)
    }
}

impl<T, U: Time> TfGen<T, U> {
    /// Evaluate the transfer function.
    ///
    /// # Arguments
    ///
    /// * `s` - Value at which the transfer function is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tf};
    /// use automatica::num_complex::Complex as C;
    /// let tf = Tf::new(poly!(1., 2., 3.), poly!(-4., -3., 1.));
    /// assert_eq!(-8.5, tf.eval(&3.));
    /// assert_eq!(C::new(0.64, -0.98), tf.eval(&C::new(0., 2.0_f32)));
    /// ```
    pub fn eval<'a, N>(&'a self, s: &'a N) -> N
    where
        T: 'a,
        N: 'a + Add<&'a T, Output = N> + Div<Output = N> + Mul<&'a N, Output = N> + Zero,
    {
        self.num.eval(s) / self.den.eval(s)
    }
}

/// Implementation of transfer function printing
impl<T, U> Display for TfGen<T, U>
where
    T: Display + One + PartialEq + PartialOrd + Signed + Zero,
    U: Time,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let (s_num, s_den) = if let Some(precision) = f.precision() {
            let num = format!("{poly:.prec$}", poly = self.num, prec = precision);
            let den = format!("{poly:.prec$}", poly = self.den, prec = precision);
            (num, den)
        } else {
            let num = format!("{}", self.num);
            let den = format!("{}", self.den);
            (num, den)
        };
        let length = s_num.len().max(s_den.len());
        let dash = "\u{2500}".repeat(length);
        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, Continuous, Discrete};
    use num_complex::Complex;
    use proptest::prelude::*;

    #[test]
    fn transfer_function_creation() {
        let num = poly!(1., 2., 3.);
        let den = poly!(-4.2, -3.12, 0.0012);
        let tf = TfGen::<_, Continuous>::new(num.clone(), den.clone());
        assert_eq!(&num, tf.num());
        assert_eq!(&den, tf.den());
    }

    #[test]
    fn relative_degree() {
        let tfz = TfGen::<_, Continuous>::new(poly!(1., 2.), poly!(-4., 6., -2.));
        let expected = tfz.relative_degree();
        assert_eq!(expected, 1);
        assert_eq!(tfz.inv().relative_degree(), -1);
        assert_eq!(
            -1,
            TfGen::<_, Continuous>::new(poly!(1., 1.), Poly::zero()).relative_degree()
        );
        assert_eq!(
            1,
            TfGen::<_, Continuous>::new(Poly::zero(), poly!(1., 1.)).relative_degree()
        );
        assert_eq!(
            0,
            TfGen::<f32, Continuous>::new(Poly::zero(), Poly::zero()).relative_degree()
        );
    }

    #[test]
    fn evaluation() {
        let tf = TfGen::<_, Continuous>::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));
        let res = tf.eval(&Complex::new(0., 0.9));
        assert_abs_diff_eq!(0.429, res.re, epsilon = 0.001);
        assert_abs_diff_eq!(1.073, res.im, epsilon = 0.001);
    }

    #[test]
    fn evaluation_by_value() {
        let tf = TfGen::<_, Continuous>::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));
        let res1 = tf.eval(&Complex::new(0., 0.9));
        let res2 = tf.eval_by_val(Complex::new(0., 0.9));
        assert_eq!(res1, res2);
    }

    #[test]
    fn tf_inversion() {
        let num1 = poly!(1., 2., 3.);
        let den1 = poly!(-4.2, -3.12, 0.0012);
        let tf1 = TfGen::<_, Discrete>::new(num1, den1);
        let num2 = poly!(-4.2, -3.12, 0.0012);
        let den2 = poly!(1., 2., 3.);
        let mut tf2 = TfGen::new(num2, den2);

        assert_eq!(tf2, (&tf1).inv());
        tf2.inv_mut();
        assert_eq!(tf2, tf1);

        assert_eq!(tf2.inv(), tf1.inv());
    }

    #[test]
    fn poles() {
        let tf = TfGen::<_, Continuous>::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(Some(vec![2., 3.]), tf.real_poles());
    }

    #[test]
    fn complex_poles() {
        use num_complex::Complex32;
        let tf = TfGen::<_, Continuous>::new(poly!(1.), poly!(10., -6., 1.));
        assert_eq!(
            vec![Complex32::new(3., -1.), Complex32::new(3., 1.)],
            tf.complex_poles()
        );
    }

    #[test]
    fn zeros() {
        let tf = TfGen::<_, Discrete>::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(None, tf.real_zeros());
    }

    #[test]
    fn complex_zeros() {
        use num_complex::Complex32;
        let tf = TfGen::<_, Discrete>::new(poly!(3.25, 3., 1.), poly!(10., -3., 1.));
        assert_eq!(
            vec![Complex32::new(-1.5, -1.), Complex32::new(-1.5, 1.)],
            tf.complex_zeros()
        );
    }

    proptest! {
        #[test]
        fn qc_tf_negative_feedback(b: f64) {
            let l = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b, 1.));
            let g = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b + 1., 1.));
            assert_eq!(g, l.feedback_n());
        }
    }

    proptest! {
    #[test]
        fn qc_tf_positive_feedback(b: f64) {
            let l = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b, 1.));
            let g = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b - 1., 1.));
            assert_eq!(g, l.feedback_p());
        }
    }

    #[test]
    fn tf_neg() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = TfGen::<_, Discrete>::new(poly!(-1., -2.), poly!(1., 5.));
        assert_eq!(-&tf1, tf2);
        assert_eq!(tf1, -(-(&tf1)));
    }

    #[test]
    fn add_references() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 5.));
        let actual = &tf1 + &tf2;
        let expected = TfGen::new(poly!(4., 2.), poly!(1., 5.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected + &TfGen::zero());
        assert_eq!(expected, &TfGen::zero() + &expected);
    }

    #[test]
    fn add_values() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(3., -4.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 5.));
        let actual = tf1 + tf2;
        let expected = TfGen::new(poly!(10., -5., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() + TfGen::zero());
        assert_eq!(expected, TfGen::zero() + expected.clone());
    }

    #[test]
    fn add_multiple_values() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(3., -4.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 5.));
        let tf3 = TfGen::new(poly!(0., 4.), poly!(3., 11., -20.));
        let actual = &(&tf1 + &tf2) + &tf3;
        let expected = TfGen::new(poly!(10., -1., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);

        let actual2 = (tf1 + tf2) + tf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn add_scalar_value() {
        let tf = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(3., -4.));
        let actual = tf + 1.;
        let expected = TfGen::new(poly!(4., -2.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn add_scalar_reference() {
        let tf = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(3., -4.));
        let actual = tf + &2.;
        let expected = TfGen::new(poly!(7., -6.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn sub_references() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(-1., 9.), poly!(4., -1.));
        let tf2 = TfGen::new(poly!(3.), poly!(4., -1.));
        let actual = &tf1 - &tf2;
        let expected = TfGen::new(poly!(-4., 9.), poly!(4., -1.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected - &TfGen::zero());
        assert_eq!(-&expected, &TfGen::zero() - &expected);
    }

    #[test]
    fn sub_values() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., -2.), poly!(4., -4.));
        let tf2 = TfGen::new(poly!(2.), poly!(2., 3.));
        let actual = tf1 - tf2;
        let expected = TfGen::new(poly!(-6., 7., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() - TfGen::zero());
        assert_eq!(-&expected, TfGen::zero() - expected);
    }

    #[test]
    fn sub_multiple_values() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., -2.), poly!(4., -4.));
        let tf2 = TfGen::new(poly!(2.), poly!(2., 3.));
        let tf3 = TfGen::new(poly!(0., 2.), poly!(8., 4., -12.));
        let actual = &(&tf1 - &tf2) - &tf3;
        let expected = TfGen::new(poly!(-6., 5., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);

        let actual2 = (tf1 - tf2) - tf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn mul_references() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 6., 5.));
        let actual = &tf1 * &tf2;
        let expected = TfGen::new(poly!(3., 6., 9.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(TfGen::zero(), &expected * &TfGen::zero());
        assert_eq!(TfGen::zero(), &TfGen::zero() * &expected);
    }

    #[test]
    fn mul_values() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = tf1 * tf2;
        let expected = TfGen::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(TfGen::zero(), expected.clone() * TfGen::zero());
        assert_eq!(TfGen::zero(), TfGen::zero() * expected);
    }

    #[test]
    fn mul_value_reference() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = tf1 * &tf2;
        let expected = TfGen::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(TfGen::zero(), expected.clone() * &TfGen::zero());
        assert_eq!(TfGen::zero(), TfGen::zero() * &expected);
    }

    #[test]
    fn div_values() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 6., 5.));
        let actual = tf2 / tf1;
        let expected = TfGen::new(poly!(3., 15.), poly!(1., 8., 20., 28., 15.));
        assert_eq!(expected, actual);
        assert_eq!(TfGen::zero(), &TfGen::zero() / &expected);
        assert!((&expected / &TfGen::zero()).eval(&1.).is_infinite());
        assert!((&TfGen::<f32, Discrete>::zero() / &TfGen::zero())
            .eval(&1.)
            .is_nan());
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn div_references() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let actual = &tf1 / &tf1;
        let expected = TfGen::new(poly!(1., 7., 13., 15.), poly!(1., 7., 13., 15.));
        assert_eq!(expected, actual);
        assert_eq!(TfGen::zero(), TfGen::zero() / expected.clone());
        assert!((expected / TfGen::zero()).eval(&1.).is_infinite());
        assert!((TfGen::<f32, Discrete>::zero() / TfGen::zero())
            .eval(&1.)
            .is_nan());
    }

    #[test]
    fn zero_tf() {
        assert!(TfGen::<f32, Continuous>::zero().is_zero());
        assert!(!TfGen::<_, Discrete>::new(poly!(0.), poly!(0.)).is_zero());
    }

    #[test]
    fn print() {
        let tf = TfGen::<_, Continuous>::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        assert_eq!(
            "1\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n1 +1s",
            format!("{}", tf)
        );

        let tf2 = TfGen::<_, Continuous>::new(poly!(1.123), poly!(0.987, -1.321));
        assert_eq!(
            "1.12\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n0.99 -1.32s",
            format!("{:.2}", tf2)
        );
    }

    #[test]
    fn normalization() {
        let tfz = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(-4., 6., -2.));
        let expected = TfGen::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz.normalize());

        let tfz2 = TfGen::<_, Discrete>::new(poly!(1.), poly!(0.));
        assert_eq!(tfz2, tfz2.normalize());
    }

    #[test]
    fn normalization_mutable() {
        let mut tfz = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(-4., 6., -2.));
        tfz.normalize_mut();
        let expected = TfGen::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz);

        let mut tfz2 = TfGen::<_, Discrete>::new(poly!(1.), poly!(0.));
        let tfz3 = tfz2.clone();
        tfz2.normalize_mut();
        assert_eq!(tfz2, tfz3);
    }

    #[test]
    fn failed_conversion_from_ss() {
        let ss = crate::Ss::new_from_slice(
            2,
            2,
            1,
            &[1., 1., 1., 1.],
            &[1., 1., 1., 1.],
            &[1., 1.],
            &[1., 1.],
        );
        let res = TfGen::<f32, Continuous>::new_from_siso(&ss);
        assert!(res.is_err());
    }

    #[test]
    fn eval_trasfer_function() {
        let s_num = Poly::new_from_coeffs(&[-1., 1.]);
        let s_den = Poly::new_from_coeffs(&[0., 1.]);
        let s = TfGen::<f64, Continuous>::new(s_num, s_den);
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        let r = p.eval(&s);
        let expected = TfGen::<f64, Continuous>::new(poly!(3., -8., 6.), poly!(0., 0., 1.));
        assert_eq!(expected, r);
    }
}
