//! # Polynomials
//!
//! Polynomial implementation
//! * builder from coefficients or roots
//! * degree
//! * extend by adding 0 coefficients to higher order terms
//! * arithmetic operations between polynomials (addition, subtraction,
//!   multiplication, division, reminder, negation)
//! * arithmetic operations with floats (addition, subtraction,
//!   multiplication, division)
//! * transformation to monic form
//! * roots finding (real and complex) using eigenvalues of the companion matrix
//! * differentiation and integration
//! * evaluation using real or complex numbers
//! * coefficient indexing
//! * zero and unit polynomials

pub mod arithmetic;
mod convex_hull;
mod fft;
mod roots;

use num_complex::Complex;
use num_traits::{Float, NumCast, One, Signed, Zero};

use std::{
    fmt::{Debug, Formatter},
    ops::{Add, Div, Index, IndexMut, Mul, Neg},
};

use crate::iterator;

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// `p(x) = c0 + c1*x + c2*x^2 + ...`
#[derive(Debug, PartialEq, Clone)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

/// Macro shortcut to crate a polynomial from its coefficients.
///
/// # Example
/// ```
/// #[macro_use] extern crate automatica;
/// use automatica::polynomial::Poly;
/// let p1 = poly!(1, 2, 3);
/// let p2 = Poly::new_from_coeffs(&[1, 2, 3]);
/// assert_eq!(p1, p2);
/// ```
#[macro_export]
macro_rules! poly {
    ($($c:expr),+ $(,)*) => {
        $crate::polynomial::Poly::new_from_coeffs(&[$($c,)*]);
    };
}

impl<T> Poly<T> {
    /// Length of the polynomial coefficients
    fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Return the coefficients of the polynomial as a slice
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let c = &[1., 2., 3.];
    /// let p = Poly::new_from_coeffs(c);
    /// assert_eq!(c, p.as_slice());
    /// ```
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T: Clone + PartialEq + Zero> Poly<T> {
    /// Create a new polynomial given a slice of real coefficients.
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// ```
    pub fn new_from_coeffs(coeffs: &[T]) -> Self {
        let mut p = Self {
            coeffs: coeffs.into(),
        };
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Create a new polynomial given a iterator of real coefficients.
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - iterator of coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs_iter(1..4);
    /// ```
    pub fn new_from_coeffs_iter<II>(coeffs: II) -> Self
    where
        II: IntoIterator<Item = T>,
    {
        let mut p = Self {
            coeffs: coeffs.into_iter().collect(),
        };
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Trim the zeros coefficients of high degree terms.
    /// It will not leave an empty `coeffs` vector: zero poly is returned.
    fn trim(&mut self) {
        // TODO try to use assert macro.
        //.rposition(|&c| relative_ne!(c, 0.0, epsilon = epsilon, max_relative = max_relative))
        if let Some(p) = self.coeffs.iter().rposition(|c| c != &T::zero()) {
            let new_length = p + 1;
            debug_assert!(new_length > 0);
            self.coeffs.truncate(new_length);
        } else {
            self.coeffs.resize(1, T::zero());
        }
        debug_assert!(!self.coeffs.is_empty());
    }

    /// Degree of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(Some(2), p.degree());
    /// ```
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        debug_assert!(
            !self.coeffs.is_empty(),
            "Degree is not defined on empty polynomial"
        );
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Extend the polynomial coefficients with 0 to the given degree in place.
    /// It does not truncate the polynomial.
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the new highest coefficient.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let mut p = Poly::new_from_coeffs(&[1, 2, 3]);
    /// p.extend(5);
    /// assert_eq!(vec![1, 2, 3, 0, 0, 0], p.coeffs());
    /// ```
    pub fn extend(&mut self, degree: usize) {
        match self.degree() {
            None => self.coeffs.resize(degree + 1, T::zero()),
            Some(d) if degree > d => self.coeffs.resize(degree + 1, T::zero()),
            Some(_) => (),
        };
        debug_assert!(!self.coeffs.is_empty());
    }
}

impl<T: Clone + Div<Output = T> + One> Poly<T> {
    /// Return the monic polynomial and the leading coefficient.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let (p2, c) = p.monic();
    /// assert_eq!(Poly::new_from_coeffs(&[0.1, 0.2, 1.]), p2);
    /// assert_eq!(10., c);
    /// ```
    #[must_use]
    pub fn monic(&self) -> (Self, T) {
        let lc = self.leading_coeff();
        let result: Vec<_> = self.coeffs.iter().map(|x| x.clone() / lc.clone()).collect();
        let monic_poly = Self { coeffs: result };

        debug_assert!(!monic_poly.coeffs.is_empty());
        (monic_poly, lc)
    }

    /// Return the monic polynomial and the leading coefficient,
    /// it mutates the polynomial in place.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let mut p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let c = p.monic_mut();
    /// assert_eq!(Poly::new_from_coeffs(&[0.1, 0.2, 1.]), p);
    /// assert_eq!(10., c);
    /// ```
    pub fn monic_mut(&mut self) -> T {
        let lc = self.leading_coeff();
        self.div_mut(lc.clone());
        debug_assert!(!self.coeffs.is_empty());
        lc
    }
}

impl<T: Clone + One> Poly<T> {
    /// Return the leading coefficient of the polynomial.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let c = p.leading_coeff();
    /// assert_eq!(10., c);
    /// ```
    #[must_use]
    pub fn leading_coeff(&self) -> T {
        self.coeffs.last().unwrap_or(&T::one()).clone()
    }
}

impl<T: Clone + Mul<Output = T> + Neg<Output = T> + One + PartialEq + Zero> Poly<T> {
    /// Create a new polynomial given a slice of real roots
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `roots` - slice of roots
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_roots(&[1., 2., 3.]);
    /// ```
    pub fn new_from_roots(roots: &[T]) -> Self {
        let mut p = roots.iter().fold(Self::one(), |acc, r| {
            acc * Self {
                coeffs: vec![-r.clone(), T::one()],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Create a new polynomial given an iterator of real roots
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `roots` - iterator of roots
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_roots_iter((1..4));
    /// ```
    pub fn new_from_roots_iter<II>(roots: II) -> Self
    where
        II: IntoIterator<Item = T>,
    {
        let mut p = roots.into_iter().fold(Self::one(), |acc, r| {
            acc * Self {
                coeffs: vec![-r, T::one()],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }
}

impl<T: Clone + PartialEq + PartialOrd + Signed + Zero> Poly<T> {
    /// Round off to zero coefficients smaller than `atol`.
    ///
    /// # Arguments
    ///
    /// * `atol` - Absolute tolerance (should be positive)
    ///
    /// # Example
    ///```
    /// use automatica::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0.002, 1., -0.0001]);
    /// let actual = p.roundoff(0.01);
    /// let expected = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// assert_eq!(expected, actual);
    ///```
    pub fn roundoff(&self, atol: T) -> Self {
        let atol = atol.abs();
        let new_coeff = self
            .coeffs
            .iter()
            .map(|c| if c.abs() < atol { T::zero() } else { c.clone() });
        Poly::new_from_coeffs_iter(new_coeff)
    }

    /// Round off to zero coefficients smaller than `atol` in place.
    ///
    /// # Arguments
    ///
    /// * `atol` - Absolute tolerance (should be positive)
    ///
    /// # Example
    ///```
    /// use automatica::Poly;
    /// let mut p = Poly::new_from_coeffs(&[1., 0.002, 1., -0.0001]);
    /// p.roundoff_mut(0.01);
    /// let expected = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// assert_eq!(expected, p);
    ///```
    pub fn roundoff_mut(&mut self, atol: T) {
        let atol = atol.abs();
        for c in &mut self.coeffs {
            if c.abs() < atol {
                *c = T::zero()
            }
        }
        self.trim();
        debug_assert!(!self.coeffs.is_empty());
    }
}

impl<T: Clone + Mul<Output = T> + NumCast + One + PartialEq + Zero> Poly<T> {
    /// Calculate the derivative of the polynomial.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let d = p.derive();
    /// assert_eq!(Poly::new_from_coeffs(&[0., 2.]), d);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the exponent of the term (`usize`) cannot be converted
    /// to `T`.
    #[must_use]
    pub fn derive(&self) -> Self {
        if self.len() == 1 {
            return Poly::zero(); // Never empty polynomial.
        }

        let derive_coeffs: Vec<_> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c.clone() * T::from(i).unwrap())
            .collect();

        let result = Self {
            coeffs: derive_coeffs,
        };
        debug_assert!(!result.coeffs.is_empty());
        result
    }
}

impl<T: Clone + Div<Output = T> + NumCast + PartialEq + Zero> Poly<T> {
    /// Calculate the integral of the polynomial. When used with integral types
    /// it does not convert the coefficients to floats, division is between
    /// integers.
    ///
    /// # Arguments
    ///
    /// * `constant` - Integration constant
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 3.]);
    /// let d = p.integrate(5.3);
    /// assert_eq!(Poly::new_from_coeffs(&[5.3, 1., 0., 1.]), d);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the exponent of the term (`usize`) cannot be converted
    /// to `T`.
    pub fn integrate(&self, constant: T) -> Self {
        if self.is_zero() {
            // Never empty polynomial.
            return Self {
                coeffs: vec![constant],
            };
        }
        let int_coeffs: Vec<_> = std::iter::once(constant)
            .chain(
                self.coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c.clone() / T::from(i + 1).unwrap()),
            )
            .collect();
        let result = Self { coeffs: int_coeffs };
        debug_assert!(!result.coeffs.is_empty());
        result
    }
}

// Evaluate the polynomial at the given real or complex number
// impl<N, T> Eval<N> for Poly<T>
// where
//     N: Copy + MulAdd<Output = N> + NumCast + Zero,
//     T: Copy + NumCast,
// {
//     /// Evaluate the polynomial using Horner's method. The evaluation is safe
//     /// if the polynomial coefficient can be casted the type `N`.
//     ///
//     /// # Arguments
//     ///
//     /// * `x` - Value at which the polynomial is evaluated.
//     ///
//     /// # Panics
//     ///
//     /// The method panics if the conversion from `T` to type `N` fails.
//     ///
//     /// # Example
//     /// ```
//     /// use automatica::{Eval, num_complex::Complex, polynomial::Poly};
//     /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
//     /// assert_eq!(18., p.eval(3.));
//     /// assert_eq!(Complex::new(-18., 0.), p.eval(Complex::new(0., 3.)));
//     /// ```
//     fn eval_ref(&self, x: &N) -> N {
//         self.coeffs
//             .iter()
//             .rev()
//             .fold(N::zero(), |acc, &c| acc.mul_add(*x, N::from(c).unwrap()))
//     }
// }

impl<T: Clone> Poly<T> {
    /// Vector copy of the polynomial's coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(vec![1., 2., 3.], p.coeffs());
    /// ```
    #[must_use]
    pub fn coeffs(&self) -> Vec<T> {
        self.coeffs.clone()
    }

    // The current implementation relies on the ability to add type N and T.
    // When the trait MulAdd<N,T> for N=Complex<T>, mul_add may be used.
    /// Evaluate the polynomial using Horner's method.
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::{num_complex::Complex, Poly};
    /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
    /// assert_eq!(18., p.eval_by_val(3.));
    /// assert_eq!(Complex::new(-18., 0.), p.eval_by_val(Complex::new(0., 3.)));
    /// ```
    pub fn eval_by_val<U>(&self, x: U) -> U
    where
        U: Add<T, Output = U> + Clone + Mul<U, Output = U> + Zero,
    {
        self.coeffs
            .iter()
            .rev()
            .fold(U::zero(), |acc, c| acc * x.clone() + c.clone())
    }
}

impl<T> Poly<T> {
    /// Evaluate the polynomial using Horner's method.
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::{num_complex::Complex, Poly};
    /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
    /// assert_eq!(18., p.eval(&3.));
    /// assert_eq!(Complex::new(-18., 0.), p.eval(&Complex::new(0., 3.)));
    /// ```
    pub fn eval<'a, U>(&'a self, x: &'a U) -> U
    where
        T: 'a,
        U: 'a + Add<&'a T, Output = U> + Mul<&'a U, Output = U> + Zero,
    {
        // Both the polynomial and the input value must have the same lifetime.
        self.coeffs
            .iter()
            .rev()
            .fold(U::zero(), |acc, c| acc * x + c)
    }
}

impl<T: Float> Poly<T> {
    /// Evaluate the ratio between to polynomials at the given value.
    /// This implementation avoids overflow issues when evaluating the
    /// numerator and the denominator separately.
    ///
    /// # Arguments
    ///
    /// * `numerator` - numerator of the polynomial ratio.
    /// * `denominator` - denominator of the polynomial ratio.
    /// * `x` - Value at which the polynomial ratio is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::Poly;
    /// let p1 = Poly::new_from_coeffs(&[4., 5., 1.]);
    /// let p2 = Poly::new_from_coeffs(&[1., 2., 3., 1.]);
    /// let x = -1e30_f32;
    /// let r = Poly::eval_poly_ratio(&p1, &p2, x);
    /// let naive = p1.eval(&x) / p2.eval(&x);
    /// assert!(naive.is_nan());
    /// assert!((0.- r).abs() < 1e-16);
    /// ```
    pub fn eval_poly_ratio(numerator: &Self, denominator: &Self, x: T) -> T {
        // When the `x` value is greater than one evaluate the polynomial ratio
        // at `1/x` reversing the coefficients.
        if x.abs() <= T::one() {
            let n = numerator.eval_by_val(x);
            let d = denominator.eval_by_val(x);
            n / d
        } else {
            let x = x.recip();
            // Zip and extend the smaller polynomial with zeros.
            // Evaluate the reversed polynomial.
            let (n, d) = iterator::zip_longest(&numerator.coeffs, &denominator.coeffs, &T::zero())
                .fold((T::zero(), T::zero()), |acc, c| {
                    (acc.0 * x + *c.0, acc.1 * x + *c.1)
                });
            n / d
        }
    }
}

/// Implement read only indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// assert_eq!(2, p[2]);
/// ```
impl<T> Index<usize> for Poly<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.coeffs[i]
    }
}

/// Implement mutable indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let mut p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// p[2] = 4;
/// assert_eq!(4, p[2]);
/// ```
impl<T> IndexMut<usize> for Poly<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.coeffs[i]
    }
}

/// Implementation of the additive identity for polynomials
///
/// # Example
/// ```
/// use automatica::{num_traits::Zero, polynomial::Poly};
/// let zero = Poly::<u8>::zero();
/// assert!(zero.is_zero());
/// ```
impl<T: Clone + PartialEq + Zero> Zero for Poly<T> {
    fn zero() -> Self {
        // The polynomial is never empty.
        Self {
            coeffs: vec![T::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0] == T::zero()
    }
}

/// Implementation of the multiplicative identity for polynomials
///
/// # Example
/// ```
/// use automatica::{num_traits::One, polynomial::Poly};
/// let one = Poly::<u8>::one();
/// assert!(one.is_one());
/// ```
impl<T: Clone + Mul<Output = T> + One + PartialEq + Zero> One for Poly<T> {
    fn one() -> Self {
        // The polynomial is never empty.
        Self {
            coeffs: vec![T::one()],
        }
    }

    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0] == T::one()
    }
}

/// Implement printing of polynomial
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let p = Poly::new_from_coeffs(&[0, 1, 2, 0, 3]);
/// assert_eq!("1s +2s^2 +3s^4", format!("{}", p));
/// ```

macro_rules! display {
    ($trait:path) => {
        impl<T: $trait + PartialOrd + Zero> $trait for Poly<T> {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                debug_assert!(!self.coeffs.is_empty());
                if self.len() == 1 {
                    return self[0].fmt(f);
                }

                let iter = self
                    .coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| !x.is_zero())
                    .enumerate();
                for (i, (n, c)) in iter {
                    match (i, f.sign_plus(), c < &T::zero()) {
                        (0, _, _) => (),
                        (_, false, false) => write!(f, " +")?,
                        (_, _, _) => write!(f, " ")?,
                    }
                    if n == 0 {
                        c.fmt(f)?;
                    } else if n == 1 {
                        c.fmt(f)?;
                        write!(f, "s")?;
                    } else {
                        c.fmt(f)?;
                        write!(f, "s^")?;
                        write!(f, "{}", n)?;
                    }
                }
                write!(f, "")
            }
        }
    };
}

display!(std::fmt::Binary);
display!(std::fmt::Display);
display!(std::fmt::LowerExp);
display!(std::fmt::LowerHex);
display!(std::fmt::Octal);
display!(std::fmt::UpperExp);
display!(std::fmt::UpperHex);

// TODO: this trait implementation works from Rust 1.41.
// It is similar to the method .coeffs().
// I keep it commented if the will be more features that require newer
// compiler version I will decomment it.
// /// Conversion from `Poly` to a `Vec` containing its coefficients.
// impl<T> From<Poly<T>> for Vec<T> {
//     fn from(poly: Poly<T>) -> Self {
//         poly.coeffs
//     }
// }

/// View the `Poly` coefficients as slice.
impl<T> AsRef<[T]> for Poly<T> {
    fn as_ref(&self) -> &[T] {
        self.coeffs.as_ref()
    }
}

/// Calculate the complex roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
///
/// # Example
///```
/// use automatica::{num_complex::Complex, polynomial};
/// let actual = polynomial::complex_quadratic_roots(0., 1.);
/// assert_eq!((-Complex::i(), Complex::i()), actual);
///```
pub fn complex_quadratic_roots<T: Float>(b: T, c: T) -> (Complex<T>, Complex<T>) {
    roots::complex_quadratic_roots_impl(b, c)
}

/// Calculate the real roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
///
/// # Example
///```
/// use automatica::polynomial;
/// let actual = polynomial::real_quadratic_roots(-2., 1.);
/// assert_eq!(Some((1., 1.)), actual);
///```
pub fn real_quadratic_roots<T: Float>(b: T, c: T) -> Option<(T, T)> {
    roots::real_quadratic_roots_impl(b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use proptest::prelude::*;

    #[test]
    fn poly_formatting() {
        assert_eq!("0", format!("{}", Poly::<i16>::zero()));
        assert_eq!("0", format!("{}", Poly::<u16>::new_from_coeffs(&[])));
        assert_eq!("1 +2s^3 -4s^4", format!("{}", poly!(1, 0, 0, 2, -4)));
        assert_eq!("1.235", format!("{:.3}", Poly::new_from_coeffs(&[1.23456])));
        let p = poly!(1.2345, -5.4321, 13.1234);
        assert_eq!("+1.23 -5.43s +13.12s^2", format!("{:+.2}", &p));
        assert_eq!("1.23 -5.43s +13.12s^2", format!("{:.2}", &p));
        assert_eq!("1.2345e0 -5.4321e0s +1.31234e1s^2", format!("{:e}", &p));
    }

    #[test]
    fn poly_creation_coeffs() {
        let c = [4.3, 5.32];
        for (c1, c2) in c.iter().zip(Poly::new_from_coeffs(&c).coeffs) {
            assert_relative_eq!(*c1, c2);
        }

        let c2 = [0., 1., 1., 0., 0., 0.];
        for (i, c) in c2[..3].iter().zip(Poly::new_from_coeffs(&c2).coeffs) {
            assert_relative_eq!(*i, c);
        }

        let zero: [f64; 1] = [0.];
        for (z, c) in zero.iter().zip(poly!(0., 0.).coeffs) {
            assert_relative_eq!(*z, c);
        }

        let int = [1, 2, 3, 4, 5];
        assert_eq!(int, Poly::new_from_coeffs(&int).coeffs.as_slice());

        let float = [0.1_f32, 0.34, 3.43];
        for (f, c) in float.iter().zip(Poly::new_from_coeffs(&float).coeffs) {
            assert_relative_eq!(*f, c);
        }

        assert_eq!(
            Poly::new_from_coeffs(&[1, 2, 3]),
            Poly::new_from_coeffs_iter(1..=3)
        );
    }

    #[test]
    fn coeffs() {
        let int = [1, 2, 3, 4, 5];
        let p = Poly::new_from_coeffs(&int);
        assert_eq!(int, p.coeffs().as_slice());
    }

    #[test]
    fn as_slice() {
        let int = [1, 2, 3, 4, 5];
        let p = Poly::new_from_coeffs(&int);
        assert_eq!(int, p.as_slice());
    }

    #[test]
    fn poly_creation_roots() {
        assert_eq!(poly!(4., 4., 1.), Poly::new_from_roots(&[-2., -2.]));

        assert_eq!(poly!(4, 4, 1), Poly::new_from_roots(&[-2, -2]));

        assert!(vec![-2., -2.]
            .iter()
            .zip(
                Poly::new_from_roots(&[-2., -2.])
                    .real_roots()
                    .unwrap()
                    .iter()
            )
            .map(|(x, y): (&f64, &f64)| (x - y).abs())
            .all(|x| x < 0.000_001));

        assert!(vec![1.0_f32, 2., 3.]
            .iter()
            .zip(
                Poly::new_from_roots(&[1., 2., 3.])
                    .real_roots()
                    .unwrap()
                    .iter()
            )
            .map(|(x, y): (&f32, &f32)| (x - y).abs())
            .all(|x| x < 0.000_01));

        assert_eq!(
            poly!(0., -2., 1., 1.),
            Poly::new_from_roots(&[-0., -2., 1.])
        );

        let v = vec![1, 2, -3];
        assert_eq!(Poly::new_from_roots(&v), Poly::new_from_roots_iter(v));
    }

    #[test]
    fn len() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(3, p.len());
    }

    #[test]
    fn degree() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(Some(2), p.degree());

        let p2 = Poly::new_from_coeffs(&[0.]);
        assert_eq!(None, p2.degree());
    }

    #[test]
    fn extend_less() {
        let mut p1 = poly!(3, 4, 2);
        let p2 = p1.clone();
        p1.extend(1);
        assert_eq!(p1, p2);
    }

    #[test]
    fn extend_more() {
        let mut p1 = poly!(3, 4, 2);
        let p2 = Poly {
            coeffs: vec![3, 4, 2, 0, 0, 0, 0],
        };
        p1.extend(6);
        assert_eq!(p1, p2);
    }

    #[test]
    fn extend_zero() {
        let mut p1 = Poly::<u32>::zero();
        let p2 = Poly {
            coeffs: vec![0, 0, 0, 0],
        };
        p1.extend(3);
        assert_eq!(p1, p2);
    }

    #[test]
    fn poly_eval() {
        let p = poly!(1., 2., 3.);
        assert_abs_diff_eq!(86., p.eval(&5.), epsilon = 0.);

        assert_abs_diff_eq!(0., Poly::<f64>::zero().eval(&6.4), epsilon = 0.);

        let p2 = poly!(3, 4, 1);
        assert_eq!(143, p2.eval(&10));
    }

    #[test]
    fn poly_cmplx_eval() {
        let p = poly!(1., 1., 1.);
        let c = Complex::new(1.0, 1.0);
        let res = Complex::new(2.0, 3.0);
        assert_eq!(res, p.eval(&c));

        assert_eq!(
            Complex::zero(),
            Poly::<f64>::new_from_coeffs(&[]).eval(&Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_eval_by_value() {
        let p = poly!(1., 2., 3.);
        let r1 = p.eval_by_val(0.);
        let r2 = p.eval(&0.);
        assert_relative_eq!(r1, r2);
    }

    #[test]
    fn eval_poly_of_poly() {
        let s = poly!(-1, 1);
        let p = poly!(1, 2, 3);
        let r = p.eval(&s);
        assert_eq!(poly!(2, -4, 3), r);
    }

    #[test]
    fn poly_ratio_evaluation() {
        let p1 = poly!(1., 2., 3.);
        let p2 = poly!(4., 5.);
        let x = 2.;
        let r1 = Poly::eval_poly_ratio(&p1, &p2, x);
        assert_relative_eq!(p1.eval(&x) / p2.eval(&x), r1);

        let y = 0.5;
        let r2 = Poly::eval_poly_ratio(&p1, &p2, y);
        assert_relative_eq!(p1.eval(&y) / p2.eval(&y), r2);
    }

    #[test]
    fn poly_ratio_overflow() {
        let p1 = Poly::new_from_coeffs(&[4., 5., 1.]);
        let p2 = Poly::new_from_coeffs(&[1., 2., 3., 1.]);
        let x = -1e30_f32;
        let r = Poly::eval_poly_ratio(&p1, &p2, x);
        let naive = p1.eval(&x) / p2.eval(&x);
        assert!(naive.is_nan());
        assert!((0. - r).abs() < 1e-16);
    }

    #[test]
    fn indexing() {
        assert_abs_diff_eq!(3., poly!(1., 3.)[1], epsilon = 0.);

        let mut p = Poly::new_from_roots(&[1., 4., 5.]);
        p[2] = 3.;
        assert_eq!(poly!(-20., 29., 3., 1.), p);
    }

    #[test]
    fn derive() {
        let p = poly!(1_u8, 2, 4, 8, 16);
        let p_prime = poly!(2_u8, 8, 24, 64);
        assert_eq!(p_prime, p.derive());
    }

    #[test]
    fn integrate() {
        let p = poly!(1_u8, 2, 4, 8, 16);
        let p2 = poly!(9_u8, 1, 1, 1, 2, 3);
        // Integer division.
        assert_eq!(p2, p.integrate(9));
    }

    #[test]
    fn derive_integrate() {
        let d = poly!(1.3, 3.5, -2.3, -1.6);
        let i = d.integrate(3.2);
        assert_eq!(d, i.derive());
    }

    #[test]
    fn float_coeffs_identities() {
        assert!(Poly::<f64>::zero().is_zero());
        assert!(Poly::<f64>::one().is_one());

        assert!(Poly::<f32>::zero().is_zero());
        assert!(Poly::<f32>::one().is_one());
    }

    #[test]
    fn integer_coeffs_identities() {
        assert!(Poly::<i8>::zero().is_zero());
        assert!(Poly::<i8>::one().is_one());

        assert!(Poly::<u8>::zero().is_zero());
        assert!(Poly::<u8>::one().is_one());

        assert!(Poly::<i16>::zero().is_zero());
        assert!(Poly::<i16>::one().is_one());

        assert!(Poly::<u16>::zero().is_zero());
        assert!(Poly::<u16>::one().is_one());

        assert!(Poly::<i32>::zero().is_zero());
        assert!(Poly::<i32>::one().is_one());

        assert!(Poly::<u32>::zero().is_zero());
        assert!(Poly::<u32>::one().is_one());

        assert!(Poly::<i64>::zero().is_zero());
        assert!(Poly::<i64>::one().is_one());

        assert!(Poly::<u64>::zero().is_zero());
        assert!(Poly::<u64>::one().is_one());

        assert!(Poly::<i128>::zero().is_zero());
        assert!(Poly::<i128>::one().is_one());

        assert!(Poly::<u128>::zero().is_zero());
        assert!(Poly::<u128>::one().is_one());

        assert!(Poly::<isize>::zero().is_zero());
        assert!(Poly::<isize>::one().is_one());

        assert!(Poly::<usize>::zero().is_zero());
        assert!(Poly::<usize>::one().is_one());
    }

    proptest! {
        #[test]
        fn qc_leading_coefficient(c: i8) {
            prop_assume!(c != 0);
            assert_eq!(c, poly!(1, -5, c).leading_coeff());
        }
    }

    #[test]
    fn monic_poly() {
        let p = poly!(-3., 6., 9.);
        let (p2, c) = p.monic();
        assert_relative_eq!(9., c);
        assert_relative_eq!(1., p2.leading_coeff());
    }

    #[test]
    fn monic_mutable_poly() {
        let mut p = poly!(-3., 6., 9.);
        let c = p.monic_mut();
        assert_relative_eq!(9., c);
        assert_relative_eq!(1., p.leading_coeff());
    }

    #[test]
    fn conversion_into_slice() {
        assert_eq!(&[3, -6, 8], poly!(3, -6, 8).as_ref());
    }

    #[test]
    fn round_off_coefficients() {
        let p = Poly::new_from_coeffs(&[1., 0.002, 1., -0.0001]);
        let actual = p.roundoff(0.01);
        let expected = Poly::new_from_coeffs(&[1., 0., 1.]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn round_off_zero() {
        let zero = Poly::zero();
        assert_eq!(zero, zero.roundoff(0.001));
    }

    #[test]
    fn round_off_returns_zero() {
        let p = Poly::new_from_coeffs(&[0.0032, 0.002, -0.0023, -0.0001]);
        let actual = p.roundoff(0.01);
        assert_eq!(Poly::zero(), actual);
    }

    #[test]
    fn round_off_coefficients_mut() {
        let mut p = Poly::new_from_coeffs(&[1., 0.002, 1., -0.0001]);
        p.roundoff_mut(0.01);
        let expected = Poly::new_from_coeffs(&[1., 0., 1.]);
        assert_eq!(expected, p);
    }
}

mod compile_fail_test {
    /// ```compile_fail
    /// use automatica::{poly, Eval};
    /// let p = poly!(1.0e200, 2., 3.);
    /// p.eval(5.0_f32);
    /// ```
    #[allow(dead_code)]
    fn a() {}

    /// ``` compile_fail
    /// use automatica::{poly, Eval};
    /// let p = poly!(1.5, 2., 3.);
    /// assert_eq!(86, p.eval(5));
    /// ```
    #[allow(dead_code)]
    fn b() {}
}
