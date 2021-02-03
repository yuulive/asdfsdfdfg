//! Arithmetic module for rational functions

use num_traits::{Inv, One, Zero};

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{polynomial::Poly, rational_function::Rf};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly;
    use num_traits::Float;

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
}
