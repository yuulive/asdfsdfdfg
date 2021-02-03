//! Rational functions

use nalgebra::RealField;
use num_complex::Complex;
use num_traits::{Float, One, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, Div, Mul},
};

use crate::polynomial::Poly;

mod arithmetic;

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
    use num_traits::Inv;

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
