//! Rational functions

use nalgebra::RealField;
use num_complex::Complex;
use num_traits::{Float, Inv, One, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::polynomial::Poly;

/// Transfer function representation of a linear system
#[derive(Clone, Debug, PartialEq)]
pub struct Rf<T> {
    /// Transfer function numerator
    num: Poly<T>,
    /// Transfer function denominator
    den: Poly<T>,
}

/// Implementation of transfer function methods
impl<T> Rf<T> {
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
        Self { num, den }
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
}

impl<T: Clone + PartialEq + Zero> Rf<T> {
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
impl<T> Rf<T> {
    /// Compute the reciprocal of a transfer function in place.
    pub fn inv_mut(&mut self) {
        std::mem::swap(&mut self.num, &mut self.den);
    }
}

impl<T: Clone> Inv for &Rf<T> {
    type Output = Rf<T>;

    /// Compute the reciprocal of a transfer function.
    fn inv(self) -> Self::Output {
        Self::Output {
            num: self.den.clone(),
            den: self.num.clone(),
        }
    }
}

impl<T: Clone> Inv for Rf<T> {
    type Output = Self;

    /// Compute the reciprocal of a transfer function.
    fn inv(mut self) -> Self::Output {
        self.inv_mut();
        self
    }
}

impl<T: Float + RealField> Rf<T> {
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

impl<T: Clone + PartialEq + Zero> Rf<T> {
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
        }
    }
}

impl<T: Clone + PartialEq + Sub<Output = T> + Zero> Rf<T> {
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
        }
    }
}

impl<T: Clone + Div<Output = T> + One + PartialEq + Zero> Rf<T> {
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
        Self { num, den }
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

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Clone + Neg<Output = T>> Neg for &Rf<T> {
    type Output = Rf<T>;

    fn neg(self) -> Self::Output {
        Self::Output {
            num: -&self.num,
            den: self.den.clone(),
        }
    }
}

/// Implementation of transfer function negation.
/// Negative sign is transferred to the numerator.
impl<T: Clone + Neg<Output = T>> Neg for Rf<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.num = -self.num;
        self
    }
}

/// Implementation of transfer function addition
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Clone + Mul<Output = T> + One + PartialEq + Zero> Add for &Rf<T> {
    type Output = Rf<T>;

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
impl<T: Clone + One + PartialEq + Zero> Add for Rf<T> {
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
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Add<T> for Rf<T> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of transfer function addition
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Add<&T> for Rf<T> {
    type Output = Self;

    fn add(mut self, rhs: &T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of transfer function subtraction
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Clone + Neg<Output = T> + PartialEq + Sub<Output = T> + Zero + One> Sub for &Rf<T> {
    type Output = Rf<T>;

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
impl<T: Clone + Neg<Output = T> + One + PartialEq + Sub<Output = T> + Zero> Sub for Rf<T> {
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
impl<T: Clone + One + PartialEq + Zero> Mul for &Rf<T> {
    type Output = Rf<T>;

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
impl<T: Clone + One + PartialEq + Zero> Mul for Rf<T> {
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
impl<T: Clone + One + PartialEq + Zero> Mul<&Rf<T>> for Rf<T> {
    type Output = Self;

    fn mul(mut self, rhs: &Rf<T>) -> Self {
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
impl<T: Clone + One + PartialEq + Zero> Div for &Rf<T> {
    type Output = Rf<T>;

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
impl<T: Clone + One + PartialEq + Zero> Div for Rf<T> {
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

impl<T: Clone + One + PartialEq + Zero> Zero for Rf<T> {
    fn zero() -> Self {
        Self {
            num: Poly::zero(),
            den: Poly::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.num.is_zero() && !self.den.is_zero()
    }
}

impl<T: Clone> Rf<T> {
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

impl<T> Rf<T> {
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
impl<T> Display for Rf<T>
where
    T: Display + One + PartialEq + PartialOrd + Zero,
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
    use crate::poly;
    use num_complex::Complex;
    use proptest::prelude::*;

    #[test]
    fn transfer_function_creation() {
        let num = poly!(1., 2., 3.);
        let den = poly!(-4.2, -3.12, 0.0012);
        let tf = Rf::new(num.clone(), den.clone());
        assert_eq!(&num, tf.num());
        assert_eq!(&den, tf.den());
    }

    #[test]
    fn relative_degree() {
        let tfz = Rf::new(poly!(1., 2.), poly!(-4., 6., -2.));
        let expected = tfz.relative_degree();
        assert_eq!(expected, 1);
        assert_eq!(tfz.inv().relative_degree(), -1);
        assert_eq!(-1, Rf::new(poly!(1., 1.), Poly::zero()).relative_degree());
        assert_eq!(1, Rf::new(Poly::zero(), poly!(1., 1.)).relative_degree());
        assert_eq!(
            0,
            Rf::<f32>::new(Poly::zero(), Poly::zero()).relative_degree()
        );
    }

    #[test]
    fn evaluation() {
        let tf = Rf::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));
        let res = tf.eval(&Complex::new(0., 0.9));
        assert_abs_diff_eq!(0.429, res.re, epsilon = 0.001);
        assert_abs_diff_eq!(1.073, res.im, epsilon = 0.001);
    }

    #[test]
    fn evaluation_by_value() {
        let tf = Rf::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));
        let res1 = tf.eval(&Complex::new(0., 0.9));
        let res2 = tf.eval_by_val(Complex::new(0., 0.9));
        assert_eq!(res1, res2);
    }

    #[test]
    fn tf_inversion() {
        let num1 = poly!(1., 2., 3.);
        let den1 = poly!(-4.2, -3.12, 0.0012);
        let tf1 = Rf::new(num1, den1);
        let num2 = poly!(-4.2, -3.12, 0.0012);
        let den2 = poly!(1., 2., 3.);
        let mut tf2 = Rf::new(num2, den2);

        assert_eq!(tf2, (&tf1).inv());
        tf2.inv_mut();
        assert_eq!(tf2, tf1);

        assert_eq!(tf2.inv(), tf1.inv());
    }

    #[test]
    fn poles() {
        let tf = Rf::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(Some(vec![2., 3.]), tf.real_poles());
    }

    #[test]
    fn complex_poles() {
        use num_complex::Complex32;
        let tf = Rf::new(poly!(1.), poly!(10., -6., 1.));
        assert_eq!(
            vec![Complex32::new(3., -1.), Complex32::new(3., 1.)],
            tf.complex_poles()
        );
    }

    #[test]
    fn zeros() {
        let tf = Rf::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(None, tf.real_zeros());
    }

    #[test]
    fn complex_zeros() {
        use num_complex::Complex32;
        let tf = Rf::new(poly!(3.25, 3., 1.), poly!(10., -3., 1.));
        assert_eq!(
            vec![Complex32::new(-1.5, -1.), Complex32::new(-1.5, 1.)],
            tf.complex_zeros()
        );
    }

    proptest! {
        #[test]
        fn qc_tf_negative_feedback(b: f64) {
            let l = Rf::new(poly!(1.), poly!(-b, 1.));
            let g = Rf::new(poly!(1.), poly!(-b + 1., 1.));
            assert_eq!(g, l.feedback_n());
        }
    }

    proptest! {
    #[test]
        fn qc_tf_positive_feedback(b: f64) {
            let l = Rf::new(poly!(1.), poly!(-b, 1.));
            let g = Rf::new(poly!(1.), poly!(-b - 1., 1.));
            assert_eq!(g, l.feedback_p());
        }
    }

    #[test]
    fn tf_neg() {
        let tf1 = Rf::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(-1., -2.), poly!(1., 5.));
        assert_eq!(-&tf1, tf2);
        assert_eq!(tf1, -(-(&tf1)));
    }

    #[test]
    fn add_references() {
        let tf1 = Rf::new(poly!(1., 2.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let actual = &tf1 + &tf2;
        let expected = Rf::new(poly!(4., 2.), poly!(1., 5.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected + &Rf::zero());
        assert_eq!(expected, &Rf::zero() + &expected);
    }

    #[test]
    fn add_values() {
        let tf1 = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let tf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let actual = tf1 + tf2;
        let expected = Rf::new(poly!(10., -5., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() + Rf::zero());
        assert_eq!(expected, Rf::<f32>::zero() + expected.clone());
    }

    #[test]
    fn add_multiple_values() {
        let tf1 = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let tf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let tf3 = Rf::new(poly!(0., 4.), poly!(3., 11., -20.));
        let actual = &(&tf1 + &tf2) + &tf3;
        let expected = Rf::new(poly!(10., -1., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);

        let actual2 = (tf1 + tf2) + tf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn add_scalar_value() {
        let tf = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let actual = tf + 1.;
        let expected = Rf::new(poly!(4., -2.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn add_scalar_reference() {
        let tf = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let actual = tf + &2.;
        let expected = Rf::new(poly!(7., -6.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn sub_references() {
        let tf1 = Rf::new(poly!(-1., 9.), poly!(4., -1.));
        let tf2 = Rf::new(poly!(3.), poly!(4., -1.));
        let actual = &tf1 - &tf2;
        let expected = Rf::new(poly!(-4., 9.), poly!(4., -1.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected - &Rf::zero());
        assert_eq!(-&expected, &Rf::zero() - &expected);
    }

    #[test]
    fn sub_values() {
        let tf1 = Rf::new(poly!(1., -2.), poly!(4., -4.));
        let tf2 = Rf::new(poly!(2.), poly!(2., 3.));
        let actual = tf1 - tf2;
        let expected = Rf::new(poly!(-6., 7., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() - Rf::zero());
        assert_eq!(-&expected, Rf::zero() - expected);
    }

    #[test]
    fn sub_multiple_values() {
        let tf1 = Rf::new(poly!(1., -2.), poly!(4., -4.));
        let tf2 = Rf::new(poly!(2.), poly!(2., 3.));
        let tf3 = Rf::new(poly!(0., 2.), poly!(8., 4., -12.));
        let actual = &(&tf1 - &tf2) - &tf3;
        let expected = Rf::new(poly!(-6., 5., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);

        let actual2 = (tf1 - tf2) - tf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn mul_references() {
        let tf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(3.), poly!(1., 6., 5.));
        let actual = &tf1 * &tf2;
        let expected = Rf::new(poly!(3., 6., 9.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), &expected * &Rf::zero());
        assert_eq!(Rf::zero(), &Rf::zero() * &expected);
    }

    #[test]
    fn mul_values() {
        let tf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = tf1 * tf2;
        let expected = Rf::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), expected.clone() * Rf::zero());
        assert_eq!(Rf::zero(), Rf::zero() * expected);
    }

    #[test]
    fn mul_value_reference() {
        let tf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = tf1 * &tf2;
        let expected = Rf::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), expected.clone() * &Rf::zero());
        assert_eq!(Rf::zero(), Rf::zero() * &expected);
    }

    #[test]
    fn div_values() {
        let tf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = Rf::new(poly!(3.), poly!(1., 6., 5.));
        let actual = tf2 / tf1;
        let expected = Rf::new(poly!(3., 15.), poly!(1., 8., 20., 28., 15.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), &Rf::zero() / &expected);
        assert!((&expected / &Rf::zero()).eval(&1.).is_infinite());
        assert!((&Rf::<f32>::zero() / &Rf::zero()).eval(&1.).is_nan());
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn div_references() {
        let tf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let actual = &tf1 / &tf1;
        let expected = Rf::new(poly!(1., 7., 13., 15.), poly!(1., 7., 13., 15.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), Rf::zero() / expected.clone());
        assert!((expected / Rf::zero()).eval(&1.).is_infinite());
        assert!((Rf::<f32>::zero() / Rf::zero()).eval(&1.).is_nan());
    }

    #[test]
    fn zero_tf() {
        assert!(Rf::<f32>::zero().is_zero());
        assert!(!Rf::new(poly!(0.), poly!(0.)).is_zero());
    }

    #[test]
    fn print() {
        let tf = Rf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        assert_eq!(
            "1\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n1 +1s",
            format!("{}", tf)
        );

        let tf2 = Rf::new(poly!(1.123), poly!(0.987, -1.321));
        assert_eq!(
            "1.12\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n0.99 -1.32s",
            format!("{:.2}", tf2)
        );
    }

    #[test]
    fn normalization() {
        let tfz = Rf::new(poly!(1., 2.), poly!(-4., 6., -2.));
        let expected = Rf::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz.normalize());

        let tfz2 = Rf::new(poly!(1.), poly!(0.));
        assert_eq!(tfz2, tfz2.normalize());
    }

    #[test]
    fn normalization_mutable() {
        let mut tfz = Rf::new(poly!(1., 2.), poly!(-4., 6., -2.));
        tfz.normalize_mut();
        let expected = Rf::new(poly!(-0.5, -1.), poly!(2., -3., 1.));
        assert_eq!(expected, tfz);

        let mut tfz2 = Rf::new(poly!(1.), poly!(0.));
        let tfz3 = tfz2.clone();
        tfz2.normalize_mut();
        assert_eq!(tfz2, tfz3);
    }

    #[test]
    fn eval_trasfer_function() {
        let s_num = Poly::new_from_coeffs(&[-1., 1.]);
        let s_den = Poly::new_from_coeffs(&[0., 1.]);
        let s = Rf::<f64>::new(s_num, s_den);
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        let r = p.eval(&s);
        let expected = Rf::<f64>::new(poly!(3., -8., 6.), poly!(0., 0., 1.));
        assert_eq!(expected, r);
    }
}
