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

use nalgebra::{ComplexField, RealField};
use num_complex::Complex;
use num_traits::{Float, Inv, One, Signed, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    linear_system::{self, SsGen},
    polynomial::Poly,
    polynomial_matrix::{MatrixOfPoly, PolyMatrix},
    Time,
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
    /// Panics
    ///
    /// If either numerator of denominator is zero the method panics.
    #[must_use]
    pub fn new(num: Poly<T>, den: Poly<T>) -> Self {
        assert!(!num.is_zero());
        assert!(!den.is_zero());
        Self {
            num,
            den,
            time: PhantomData::<U>,
        }
    }

    /// Extract transfer function numerator
    #[must_use]
    pub fn num(&self) -> &Poly<T> {
        &self.num
    }

    /// Extract transfer function denominator
    #[must_use]
    pub fn den(&self) -> &Poly<T> {
        &self.den
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

impl<T: ComplexField + Debug + Float + RealField, U: Time> TfGen<T, U> {
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

    /// Normalization of transfer function
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
        let (den, an) = self.den.monic();
        let num = &self.num / an;
        Self {
            num,
            den,
            time: PhantomData,
        }
    }

    /// In place normalization of transfer function
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
        let an = self.den.monic_mut();
        self.num.div_mut(an);
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
            pub fn new_from_siso(ss: &SsGen<$ty, U>) -> Result<Self, &'static str> {
                let (pc, a_inv) = $laverrier(&ss.a);
                let g = a_inv.left_mul(&ss.c).right_mul(&ss.b);
                let rest = PolyMatrix::multiply(&pc, &ss.d);
                let tf = g + rest;
                if let Some(num) = MatrixOfPoly::from(tf).single() {
                    Ok(Self::new(num.clone(), pc))
                } else {
                    Err("Linear system is not Single Input Single Output")
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

/// Implementation of transfer function subtraction
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Sub for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn sub(self, rhs: Self) -> Self::Output {
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
        let num = &self.num * &rhs.num;
        let den = &self.den * &rhs.den;
        Self::Output::new(num, den)
    }
}

/// Implementation of transfer function multiplication
impl<T: Float, U: Time> Mul for TfGen<T, U> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self {
        self.num = self.num * rhs.num;
        self.den = self.den * rhs.den;
        self
    }
}

/// Implementation of transfer function division
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Float, U: Time> Div for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn div(self, rhs: Self) -> Self::Output {
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
        self.num = self.num * rhs.den;
        self.den = self.den * rhs.num;
        self
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
    /// use num_complex::Complex as C;
    /// let tf = Tf::new(poly!(1., 2., 3.), poly!(-4., -3., 1.));
    /// assert_eq!(-8.5, tf.eval1(3.));
    /// assert_eq!(C::new(0.64, -0.98), tf.eval1(C::new(0., 2.0_f32)));
    /// ```
    pub fn eval1<N>(&self, s: N) -> N
    where
        N: Add<T, Output = N> + Clone + Div<Output = N> + Mul<Output = N> + Zero,
    {
        self.num.eval1(s.clone()) / self.den.eval1(s)
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
    /// use num_complex::Complex as C;
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
impl<T: Display + One + PartialEq + Signed + Zero, U: Time> Display for TfGen<T, U> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s_num = self.num.to_string();
        let s_den = self.den.to_string();

        let length = s_num.len().max(s_den.len());
        let dash = "\u{2500}".repeat(length);

        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, Continuous, Discrete};

    #[test]
    fn transfer_function_creation() {
        let num = poly!(1., 2., 3.);
        let den = poly!(-4.2, -3.12, 0.0012);
        let tf = TfGen::<_, Continuous>::new(num.clone(), den.clone());
        assert_eq!(&num, tf.num());
        assert_eq!(&den, tf.den());
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

    #[quickcheck]
    fn tf_negative_feedback(b: f64) -> bool {
        let l = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b, 1.));
        let g = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b + 1., 1.));
        g == l.feedback_n()
    }

    #[quickcheck]
    fn tf_positive_feedback(b: f64) -> bool {
        let l = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b, 1.));
        let g = TfGen::<_, Continuous>::new(poly!(1.), poly!(-b - 1., 1.));
        g == l.feedback_p()
    }

    #[test]
    fn tf_neg() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = TfGen::<_, Discrete>::new(poly!(-1., -2.), poly!(1., 5.));
        assert_eq!(-&tf1, tf2);
        assert_eq!(tf1, -(-(&tf1)));
    }

    #[test]
    fn tf_add1() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 5.));
        let actual = &tf1 + &tf2;
        let expected = TfGen::new(poly!(4., 2.), poly!(1., 5.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_add2() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(3., -4.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 5.));
        let actual = tf1 + tf2;
        let expected = TfGen::new(poly!(10., -5., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_add3() {
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
    fn tf_sub1() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(-1., 9.), poly!(4., -1.));
        let tf2 = TfGen::new(poly!(3.), poly!(4., -1.));
        let actual = &tf1 - &tf2;
        let expected = TfGen::new(poly!(-4., 9.), poly!(4., -1.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_sub2() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., -2.), poly!(4., -4.));
        let tf2 = TfGen::new(poly!(2.), poly!(2., 3.));
        let actual = tf1 - tf2;
        let expected = TfGen::new(poly!(-6., 7., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_sub3() {
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
    fn tf_mul() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 6., 5.));
        let actual = &tf1 * &tf2;
        let expected = TfGen::new(poly!(3., 6., 9.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_mul2() {
        let tf1 = TfGen::<_, Continuous>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = tf1 * tf2;
        let expected = TfGen::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn tf_div1() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 6., 5.));
        let actual = tf2 / tf1;
        let expected = TfGen::new(poly!(3., 15.), poly!(1., 8., 20., 28., 15.));
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn tf_div2() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let actual = &tf1 / &tf1;
        let expected = TfGen::new(poly!(1., 7., 13., 15.), poly!(1., 7., 13., 15.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn print() {
        let tf = TfGen::<_, Continuous>::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        assert_eq!("1\n─────\n1 +1s", format!("{}", tf));
    }

    #[test]
    fn normalization() {
        let tfz = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(-4., 6., -2.));
        let expected = TfGen::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz.normalize());
    }

    #[test]
    fn normalization_mutable() {
        let mut tfz = TfGen::<_, Discrete>::new(poly!(1., 2.), poly!(-4., 6., -2.));
        tfz.normalize_mut();
        let expected = TfGen::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz);
    }
}
