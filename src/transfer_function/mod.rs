//! # Transfer function and matrices of transfer functions
//!
//! `Tf` contains the numerator and the denominator separately. Zeroes an poles
//! can be calculated.
//!
//! `TfMatrix` allow the definition of a matrix of transfer functions. The
//! numerators are stored in a matrix, while the denominator is stored once,
//! since it is equal for every transfer function.

pub mod continuous;
pub mod discrete;
pub mod matrix;

use nalgebra::{ComplexField, RealField, Scalar};
use num_complex::Complex;
use num_traits::{Float, Inv, MulAdd, One, Signed, Zero};

use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

use crate::{
    linear_system::{self, SsGen},
    polynomial::{matrix::MatrixOfPoly, Poly},
    Eval, Time,
};

/// Transfer function representation of a linear system
#[derive(Debug, PartialEq)]
pub struct TfGen<T, U: Time> {
    /// Transfer function numerator
    num: Poly<T>,
    /// Transfer function denominator
    den: Poly<T>,
    /// Tag to disambiguate continuous and discrete
    _type: PhantomData<U>,
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
    pub fn new(num: Poly<T>, den: Poly<T>) -> Self {
        assert!(!num.is_zero());
        assert!(!den.is_zero());
        Self {
            num,
            den,
            _type: PhantomData::<U>,
        }
    }

    /// Extract transfer function numerator
    pub fn num(&self) -> &Poly<T> {
        &self.num
    }

    /// Extract transfer function denominator
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
            _type: PhantomData::<U>,
        }
    }
}

impl<T: ComplexField + Debug + Float + RealField + Scalar, U: Time> TfGen<T, U> {
    /// Calculate the poles of the transfer function
    pub fn poles(&self) -> Option<Vec<T>> {
        self.den.roots()
    }

    /// Calculate the poles of the transfer function
    pub fn complex_poles(&self) -> Vec<Complex<T>> {
        self.den.complex_roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn zeros(&self) -> Option<Vec<T>> {
        self.num.roots()
    }

    /// Calculate the zeros of the transfer function
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
    pub fn feedback_n(&self) -> Self {
        Self {
            num: self.num.clone(),
            den: &self.den + &self.num,
            _type: PhantomData::<U>,
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
    pub fn feedback_p(&self) -> Self {
        Self {
            num: self.num.clone(),
            den: &self.den - &self.num,
            _type: PhantomData::<U>,
        }
    }
}

impl<U: Time> TryFrom<SsGen<f64, U>> for TfGen<f64, U> {
    type Error = &'static str;

    /// Convert a state-space representation into transfer functions.
    /// Conversion is available for Single Input Single Output system.
    /// If fails if the system is not SISO
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn try_from(ss: SsGen<f64, U>) -> Result<Self, Self::Error> {
        let (pc, a_inv) = linear_system::leverrier(ss.a());
        let g = a_inv.left_mul(ss.c()).right_mul(ss.b());
        let rest = pc.multiply(ss.d());
        let tf = g + rest;
        if let Some(num) = MatrixOfPoly::from(tf).siso() {
            Ok(Self::new(num.clone(), pc))
        } else {
            Err("Linear system is not Single Input Single Output")
        }
    }
}

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Float, U: Time> Neg for &TfGen<T, U> {
    type Output = TfGen<T, U>;

    fn neg(self) -> Self::Output {
        Self::Output::new(-&self.num, self.den.clone())
    }
}

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Float, U: Time> Neg for TfGen<T, U> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
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
impl<T: Float, U: Time> Add for TfGen<T, U> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Add::add(&self, &rhs)
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
impl<T: Float, U: Time> Sub for TfGen<T, U> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Sub::sub(&self, &rhs)
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

    fn mul(self, rhs: Self) -> Self {
        Mul::mul(&self, &rhs)
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
impl<T: Float, U: Time> Div for TfGen<T, U> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Div::div(&self, &rhs)
    }
}

/// Implementation of the evaluation of a transfer function with complex numbers.
impl<T: Float + MulAdd<Output = T>, U: Time> Eval<Complex<T>> for TfGen<T, U> {
    fn eval(&self, s: &Complex<T>) -> Complex<T> {
        self.num.eval(s) / self.den.eval(s)
    }
}

/// Implementation of the evaluation of a transfer function with real numbers.
impl<T: Float + MulAdd<Output = T>, U: Time> Eval<T> for TfGen<T, U> {
    fn eval(&self, s: &T) -> T {
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
        let tf1 = TfGen::<_, Discrete>::new(num1.clone(), den1.clone());
        let num2 = poly!(-4.2, -3.12, 0.0012);
        let den2 = poly!(1., 2., 3.);
        let mut tf2 = TfGen::new(num2.clone(), den2.clone());
        assert_eq!(tf2, tf1.inv());
        tf2.inv_mut();
        assert_eq!(tf2, tf1);
    }

    #[test]
    fn poles() {
        let tf = TfGen::<_, Continuous>::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(Some(vec![2., 3.]), tf.poles());
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
        assert_eq!(Some(vec![]), tf.zeros());
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
    fn tf_div() {
        let tf1 = TfGen::<_, Discrete>::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = TfGen::new(poly!(3.), poly!(1., 6., 5.));
        let actual = tf2 / tf1;
        let expected = TfGen::new(poly!(3., 15.), poly!(1., 8., 20., 28., 15.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn print() {
        let tf = TfGen::<_, Continuous>::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        assert_eq!("1\n──────\n1 +1*s", format!("{}", tf));
    }
}
