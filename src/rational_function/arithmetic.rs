//! Arithmetic module for rational functions

use num_traits::{Inv, One, Zero};

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{polynomial::Poly, rational_function::Rf};

/// Implementation of rational function methods
impl<T> Rf<T> {
    /// Compute the reciprocal of a rational function in place.
    pub fn inv_mut(&mut self) {
        std::mem::swap(&mut self.num, &mut self.den);
    }
}

impl<T: Clone> Inv for &Rf<T> {
    type Output = Rf<T>;

    /// Compute the reciprocal of a rational function.
    fn inv(self) -> Self::Output {
        Self::Output {
            num: self.den.clone(),
            den: self.num.clone(),
        }
    }
}

impl<T: Clone> Inv for Rf<T> {
    type Output = Self;

    /// Compute the reciprocal of a rational function.
    fn inv(mut self) -> Self::Output {
        self.inv_mut();
        self
    }
}

/// Implementation of rational function negation.
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

/// Implementation of rational function negation.
/// Negative sign is transferred to the numerator.
impl<T: Clone + Neg<Output = T>> Neg for Rf<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.num = -self.num;
        self
    }
}

/// Implementation of rational function addition
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

/// Implementation of rational function addition
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

/// Implementation of rational function addition
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Add<T> for Rf<T> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of rational function addition
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Add<&T> for Rf<T> {
    type Output = Self;

    fn add(mut self, rhs: &T) -> Self {
        self.num = self.num + &self.den * rhs;
        self
    }
}

/// Implementation of rational function subtraction
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

/// Implementation of rational function subtraction
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

/// Implementation of rational function multiplication
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

/// Implementation of rational function multiplication
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

/// Implementation of rational function multiplication
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

/// Implementation of rational function division
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

/// Implementation of rational function division
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
    fn rf_inversion() {
        let num1 = poly!(1., 2., 3.);
        let den1 = poly!(-4.2, -3.12, 0.0012);
        let rf1 = Rf::new(num1, den1);
        let num2 = poly!(-4.2, -3.12, 0.0012);
        let den2 = poly!(1., 2., 3.);
        let mut rf2 = Rf::new(num2, den2);

        assert_eq!(rf2, (&rf1).inv());
        rf2.inv_mut();
        assert_eq!(rf2, rf1);

        assert_eq!(rf2.inv(), rf1.inv());
    }

    #[test]
    fn rf_neg() {
        let rf1 = Rf::new(poly!(1., 2.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(-1., -2.), poly!(1., 5.));
        assert_eq!(-&rf1, rf2);
        assert_eq!(rf1, -(-(&rf1)));
    }

    #[test]
    fn add_references() {
        let rf1 = Rf::new(poly!(1., 2.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let actual = &rf1 + &rf2;
        let expected = Rf::new(poly!(4., 2.), poly!(1., 5.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected + &Rf::zero());
        assert_eq!(expected, &Rf::zero() + &expected);
    }

    #[test]
    fn add_values() {
        let rf1 = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let rf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let actual = rf1 + rf2;
        let expected = Rf::new(poly!(10., -5., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() + Rf::zero());
        assert_eq!(expected, Rf::<f32>::zero() + expected.clone());
    }

    #[test]
    fn add_multiple_values() {
        let rf1 = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let rf2 = Rf::new(poly!(3.), poly!(1., 5.));
        let rf3 = Rf::new(poly!(0., 4.), poly!(3., 11., -20.));
        let actual = &(&rf1 + &rf2) + &rf3;
        let expected = Rf::new(poly!(10., -1., 10.), poly!(3., 11., -20.));
        assert_eq!(expected, actual);

        let actual2 = (rf1 + rf2) + rf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn add_scalar_value() {
        let rf = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let actual = rf + 1.;
        let expected = Rf::new(poly!(4., -2.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn add_scalar_reference() {
        let rf = Rf::new(poly!(1., 2.), poly!(3., -4.));
        let actual = rf + &2.;
        let expected = Rf::new(poly!(7., -6.), poly!(3., -4.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn sub_references() {
        let rf1 = Rf::new(poly!(-1., 9.), poly!(4., -1.));
        let rf2 = Rf::new(poly!(3.), poly!(4., -1.));
        let actual = &rf1 - &rf2;
        let expected = Rf::new(poly!(-4., 9.), poly!(4., -1.));
        assert_eq!(expected, actual);
        assert_eq!(expected, &expected - &Rf::zero());
        assert_eq!(-&expected, &Rf::zero() - &expected);
    }

    #[test]
    fn sub_values() {
        let rf1 = Rf::new(poly!(1., -2.), poly!(4., -4.));
        let rf2 = Rf::new(poly!(2.), poly!(2., 3.));
        let actual = rf1 - rf2;
        let expected = Rf::new(poly!(-6., 7., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);
        assert_eq!(expected, expected.clone() - Rf::zero());
        assert_eq!(-&expected, Rf::zero() - expected);
    }

    #[test]
    fn sub_multiple_values() {
        let rf1 = Rf::new(poly!(1., -2.), poly!(4., -4.));
        let rf2 = Rf::new(poly!(2.), poly!(2., 3.));
        let rf3 = Rf::new(poly!(0., 2.), poly!(8., 4., -12.));
        let actual = &(&rf1 - &rf2) - &rf3;
        let expected = Rf::new(poly!(-6., 5., -6.), poly!(8., 4., -12.));
        assert_eq!(expected, actual);

        let actual2 = (rf1 - rf2) - rf3;
        assert_eq!(actual, actual2);
    }

    #[test]
    fn mul_references() {
        let rf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(3.), poly!(1., 6., 5.));
        let actual = &rf1 * &rf2;
        let expected = Rf::new(poly!(3., 6., 9.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), &expected * &Rf::zero());
        assert_eq!(Rf::zero(), &Rf::zero() * &expected);
    }

    #[test]
    fn mul_values() {
        let rf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = rf1 * rf2;
        let expected = Rf::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), expected.clone() * Rf::zero());
        assert_eq!(Rf::zero(), Rf::zero() * expected);
    }

    #[test]
    fn mul_value_reference() {
        let rf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(-5.), poly!(1., 6., 5.));
        let actual = rf1 * &rf2;
        let expected = Rf::new(poly!(-5., -10., -15.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), expected.clone() * &Rf::zero());
        assert_eq!(Rf::zero(), Rf::zero() * &expected);
    }

    #[test]
    fn div_values() {
        let rf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let rf2 = Rf::new(poly!(3.), poly!(1., 6., 5.));
        let actual = rf2 / rf1;
        let expected = Rf::new(poly!(3., 15.), poly!(1., 8., 20., 28., 15.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), &Rf::zero() / &expected);
        assert!((&expected / &Rf::zero()).eval(&1.).is_infinite());
        assert!((&Rf::<f32>::zero() / &Rf::zero()).eval(&1.).is_nan());
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn div_references() {
        let rf1 = Rf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let actual = &rf1 / &rf1;
        let expected = Rf::new(poly!(1., 7., 13., 15.), poly!(1., 7., 13., 15.));
        assert_eq!(expected, actual);
        assert_eq!(Rf::zero(), Rf::zero() / expected.clone());
        assert!((expected / Rf::zero()).eval(&1.).is_infinite());
        assert!((Rf::<f32>::zero() / Rf::zero()).eval(&1.).is_nan());
    }

    #[test]
    fn zero_rf() {
        assert!(Rf::<f32>::zero().is_zero());
        assert!(!Rf::new(poly!(0.), poly!(0.)).is_zero());
    }
}
