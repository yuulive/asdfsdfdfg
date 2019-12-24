//! # Polynomials
//!
//! `Poly` implements usual addition, subtraction and multiplication between
//! polynomials, operations with scalars are supported also.
//!
//! Methods for roots finding, companion matrix and evaluation are implemented.

pub mod matrix;

use nalgebra::{ComplexField, DMatrix, RealField, Scalar, Schur};
use num_complex::Complex;
use num_traits::{Float, MulAdd, Num, NumCast, One, Signed, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use crate::{polynomial::matrix::PolyMatrix, utils, Eval};

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// p(x) = c0 + c1*x + c2*x^2 + ...
#[derive(Debug, PartialEq, Clone)]
pub struct Poly<T> {
    pub(crate) coeffs: Vec<T>,
}

/// Macro shortcut to crate a polynomial from its coefficients.
///
/// # Example
/// ```
/// #[macro_use] extern crate automatica;
/// let p = poly!(1., 2., 3.);
/// assert_eq!(Some(2), p.degree());
/// ```
#[macro_export]
macro_rules! poly {
    ($($c:expr),+ $(,)*) => {
        $crate::polynomial::Poly::new_from_coeffs(&[$($c,)*]);
    };
}

/// Implementation methods for Poly struct
impl<T> Poly<T> {
    /// Length of the polynomial coefficients
    pub(crate) fn len(&self) -> usize {
        self.coeffs.len()
    }
}

/// Implementation methods for Poly struct
impl<T: Copy> Poly<T> {
    /// Vector of the polynomial's coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(vec![1., 2., 3.], p.coeffs());
    /// ```
    pub fn coeffs(&self) -> Vec<T> {
        self.coeffs.clone()
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Num + Zero> Poly<T> {
    /// Degree of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(Some(2), p.degree());
    /// ```
    pub fn degree(&self) -> Option<usize> {
        assert!(
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
            _ => (),
        };
    }
}

impl<T: Copy + Div<Output = T>> Poly<T> {
    /// In place division with a real number
    ///
    /// # Arguments
    ///
    /// * `d` - Real number divisor
    ///
    /// # Example
    /// ```
    /// use automatica::poly;
    /// let mut p = poly!(3, 4, 5);
    /// p.div_mut(2);
    /// assert_eq!(poly!(1, 2, 2), p);
    /// ```
    pub fn div_mut(&mut self, d: T) {
        for c in self.coeffs.iter_mut() {
            *c = *c / d;
        }
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Div<Output = T> + One> Poly<T> {
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
    pub fn monic(&self) -> (Self, T) {
        let lc = self.leading_coeff();
        let result: Vec<_> = self.coeffs.iter().map(|&x| x / lc).collect();
        let monic_poly = Self { coeffs: result };

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
        self.div_mut(lc);
        lc
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + One> Poly<T> {
    /// Return the leading coefficient of the polynomial.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let c = p.leading_coeff();
    /// assert_eq!(10., c);
    /// ```
    pub fn leading_coeff(&self) -> T {
        *self.coeffs.last().unwrap_or(&T::one())
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + PartialEq + Zero> Poly<T> {
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

    /// Trim the zeros coefficients of high degree terms.
    fn trim(&mut self) {
        // TODO try to use assert macro.
        //.rposition(|&c| relative_ne!(c, 0.0, epsilon = epsilon, max_relative = max_relative))
        if let Some(p) = self.coeffs.iter().rposition(|&c| c != T::zero()) {
            let new_length = p + 1;
            self.coeffs.truncate(new_length);
        } else {
            self.coeffs.resize(1, T::zero());
        }
    }
}

/// Implementation methods for Poly struct
impl<T: AddAssign + Copy + Num + Neg<Output = T>> Poly<T> {
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
        let mut p = roots.iter().fold(Self::one(), |acc, &r| {
            acc * Self {
                coeffs: vec![-r, T::one()],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }
}

/// Implementation methods for Poly struct
impl<T: ComplexField + Float + RealField + Scalar> Poly<T> {
    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    /// of the monic polynomial with opposite sign.
    pub(crate) fn companion(&self) -> Option<DMatrix<T>> {
        match self.degree() {
            Some(degree) if degree > 0 => {
                let hi_coeff = self.coeffs[degree];
                Some(DMatrix::from_fn(degree, degree, |i, j| {
                    if j == degree - 1 {
                        -self.coeffs[i] / hi_coeff // monic polynomial
                    } else if i == j + 1 {
                        T::one()
                    } else {
                        T::zero()
                    }
                }))
            }
            _ => None,
        }
    }

    /// Calculate the real roots of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let roots = &[0., -1., 1.];
    /// let p = Poly::new_from_roots(roots);
    /// assert_eq!(roots, p.roots().unwrap().as_slice());
    /// ```
    pub fn roots(&self) -> Option<Vec<T>> {
        if self.degree() == Some(2) {
            if let Some(r) = quadratic_roots(self[1] / self[2], self[0] / self[2]) {
                Some(vec![r.0, r.1])
            } else {
                None
            }
        } else {
            // Build the companion matrix
            let comp = match self.companion() {
                Some(comp) => comp,
                _ => return Some(vec![]),
            };
            let schur = Schur::new(comp);
            schur.eigenvalues().map(|e| e.as_slice().to_vec())
        }
    }

    /// Calculate the complex roots of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let i = num_complex::Complex::i();
    /// assert_eq!(vec![-i, i], p.complex_roots());
    /// ```
    pub fn complex_roots(&self) -> Vec<Complex<T>> {
        if self.degree() == Some(2) {
            let b = self[1] / self[2];
            let c = self[0] / self[2];
            let (r1, r2) = complex_quadratic_roots(b, c);
            vec![r1, r2]
        } else {
            let comp = match self.companion() {
                Some(comp) => comp,
                _ => return vec![],
            };
            let schur = Schur::new(comp);
            schur.complex_eigenvalues().as_slice().to_vec()
        }
    }
}

/// Calculate the complex roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
#[allow(clippy::many_single_char_names)]
pub(crate) fn complex_quadratic_roots<T: Float>(b: T, c: T) -> (Complex<T>, Complex<T>) {
    let b_ = b / T::from(2.0_f32).unwrap(); // Safe cast, it's exact.
    let d = b_.powi(2) - c; // Discriminant
    let (root1_r, root1_i, root2_r, root2_i) = if d.is_zero() {
        (-b_, T::zero(), -b_, T::zero())
    } else if d.is_sign_negative() {
        // Negative discriminant.
        let s = (-d).sqrt();
        (-b_, -s, -b_, s)
    } else {
        // Positive discriminant.
        let s = d.sqrt();
        let g = if b > T::zero() { T::one() } else { -T::one() };
        let h = -(b_ + g * s);
        (c / h, T::zero(), h, T::zero())
    };

    (
        Complex::new(root1_r, root1_i),
        Complex::new(root2_r, root2_i),
    )
}

/// Calculate the real roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
#[allow(clippy::many_single_char_names)]
pub(crate) fn quadratic_roots<T: Float>(b: T, c: T) -> Option<(T, T)> {
    let b_ = b / T::from(2.0_f32).unwrap(); // Safe cast, it's exact.
    let d = b_.powi(2) - c; // Discriminant
    let (r1, r2) = if d.is_zero() {
        (-b_, -b_)
    } else if d.is_sign_negative() {
        return None;
    } else {
        // Positive discriminant.
        let s = d.sqrt();
        let g = if b > T::zero() { T::one() } else { -T::one() };
        let h = -(b_ + g * s);
        (c / h, h)
    };

    Some((r1, r2))
}

/// Implementation methods for Poly struct
impl Poly<f64> {
    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn multiply(&self, rhs: &DMatrix<f64>) -> PolyMatrix<f64> {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

impl Poly<f32> {
    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn multiply(&self, rhs: &DMatrix<f32>) -> PolyMatrix<f32> {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Mul<Output = T> + NumCast + One> Poly<T> {
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
    pub fn derive(&self) -> Self {
        let derive_coeffs: Vec<_> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| *c * T::from(i).unwrap())
            .collect();
        Self {
            coeffs: derive_coeffs,
        }
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Div<Output = T> + NumCast> Poly<T> {
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
        let int_coeffs: Vec<_> = std::iter::once(constant)
            .chain(
                self.coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| *c / T::from(i + 1).unwrap()),
            )
            .collect();
        Self { coeffs: int_coeffs }
    }
}

/// Evaluate the polynomial at the given real or complex number
impl<N, T> Eval<N> for Poly<T>
where
    N: Copy + MulAdd<Output = N> + NumCast + Zero,
    T: Copy + NumCast,
{
    /// Evaluate the polynomial using Horner's method. The evaluation is safe
    /// if the polynomial coefficient can be casted the type `N`.
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated.
    ///
    /// # Panics
    ///
    /// The method panics if the conversion from `T` to type `N` fails.
    ///
    /// # Example
    /// ```
    /// use automatica::{Eval, polynomial::Poly};
    /// use num_complex::Complex;
    /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
    /// assert_eq!(18., p.eval(&3.));
    /// assert_eq!(Complex::new(-18., 0.), p.eval(&Complex::new(0., 3.)));
    /// ```
    fn eval(&self, x: &N) -> N {
        self.coeffs
            .iter()
            .rev()
            .fold(N::zero(), |acc, &c| acc.mul_add(*x, N::from(c).unwrap()))
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

    fn index(&self, i: usize) -> &T {
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
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.coeffs[i]
    }
}

/// Implementation of polynomial negation
impl<T: Copy + Neg<Output = T>> Neg for &Poly<T> {
    type Output = Poly<T>;

    fn neg(self) -> Self::Output {
        let c: Vec<_> = self.coeffs.iter().map(|&i| -i).collect();
        Poly { coeffs: c }
    }
}

/// Implementation of polynomial negation
impl<T: Copy + Neg<Output = T>> Neg for Poly<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
    }
}

/// Implementation of polynomial addition
impl<T: Copy + Num> Add<Poly<T>> for Poly<T> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.coeffs.iter().enumerate() {
                rhs[i] = rhs[i] + *c;
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i] + *c;
            }
            self
        };
        result.trim();
        result
    }
}

/// Implementation of polynomial addition
impl<T: Copy + Num> Add<&Poly<T>> for &Poly<T> {
    type Output = Poly<T>;

    fn add(self, rhs: &Poly<T>) -> Poly<T> {
        let new_coeffs = utils::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), Add::add);
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float addition
impl<T: Add<Output = T> + Copy> Add<T> for Poly<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        let mut result = self.clone();
        result[0] = result[0] + rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result
    }
}

macro_rules! impl_add_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Add<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn add(self, rhs: Poly<Self>) -> Poly<Self> {
                rhs + self
            }
        }
    };
}

impl_add_for_poly!(
    /// Implementation of f32 and polynomial addition
    f32
);
impl_add_for_poly!(
    /// Implementation of f64 and polynomial addition
    f64
);
impl_add_for_poly!(
    /// Implementation of i8 and polynomial addition
    i8
);
impl_add_for_poly!(
    /// Implementation of u8 and polynomial addition
    u8
);
impl_add_for_poly!(
    /// Implementation of i16 and polynomial addition
    i16
);
impl_add_for_poly!(
    /// Implementation of u16 and polynomial addition
    u16
);
impl_add_for_poly!(
    /// Implementation of i32 and polynomial addition
    i32
);
impl_add_for_poly!(
    /// Implementation of u32 and polynomial addition
    u32
);
impl_add_for_poly!(
    /// Implementation of i64 and polynomial addition
    i64
);
impl_add_for_poly!(
    /// Implementation of u64 and polynomial addition
    u64
);
impl_add_for_poly!(
    /// Implementation of i128 and polynomial addition
    i128
);
impl_add_for_poly!(
    /// Implementation of u128 and polynomial addition
    u128
);
impl_add_for_poly!(
    /// Implementation of isize and polynomial addition
    isize
);
impl_add_for_poly!(
    /// Implementation of usize and polynomial addition
    usize
);

/// Implementation of polynomial subtraction
impl<T: Copy + PartialEq + Sub<Output = T> + Zero> Sub for Poly<T> {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.len() < rhs.len() {
            // iterate on rhs and do the subtraction until self has values,
            // then invert the coefficients of rhs
            for i in 0..rhs.len() {
                rhs[i] = *self.coeffs.get(i).unwrap_or(&T::zero()) - rhs[i];
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i] - *c;
            }
            self
        };
        result.trim();
        result
    }
}

/// Implementation of polynomial subtraction
impl<T: Copy + PartialEq + Sub<Output = T> + Zero> Sub for &Poly<T> {
    type Output = Poly<T>;

    fn sub(self, rhs: Self) -> Poly<T> {
        let new_coeffs =
            utils::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), |x, y| x - y);
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float subtraction
impl<T: Copy + Sub<Output = T>> Sub<T> for Poly<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        let mut result = self.clone();
        result[0] = result[0] - rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result
    }
}

macro_rules! impl_sub_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Sub<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn sub(self, rhs: Poly<Self>) -> Poly<Self> {
                let mut result = rhs.clone();
                // Non need for trimming since the addition of a float doesn't
                // modify the coefficients of order higher than zero.
                result[0] = self - result[0];
                result
            }
        }
    };
}

impl_sub_for_poly!(
    /// Implementation of f32 and polynomial subtraction
    f32
);
impl_sub_for_poly!(
    /// Implementation of f64 and polynomial subtraction
    f64
);
impl_sub_for_poly!(
    /// Implementation of i8 and polynomial subtraction
    i8
);
impl_sub_for_poly!(
    /// Implementation of u8 and polynomial subtraction
    u8
);
impl_sub_for_poly!(
    /// Implementation of i16 and polynomial subtraction
    i16
);
impl_sub_for_poly!(
    /// Implementation of u16 and polynomial subtraction
    u16
);
impl_sub_for_poly!(
    /// Implementation of i32 and polynomial subtraction
    i32
);
impl_sub_for_poly!(
    /// Implementation of u32 and polynomial subtraction
    u32
);
impl_sub_for_poly!(
    /// Implementation of i64 and polynomial subtraction
    i64
);
impl_sub_for_poly!(
    /// Implementation of u64 and polynomial subtraction
    u64
);
impl_sub_for_poly!(
    /// Implementation of i128 and polynomial subtraction
    i128
);
impl_sub_for_poly!(
    /// Implementation of u128 and polynomial subtraction
    u128
);
impl_sub_for_poly!(
    /// Implementation of isize and polynomial subtraction
    isize
);
impl_sub_for_poly!(
    /// Implementation of usize and polynomial subtraction
    usize
);

/// Implementation of polynomial multiplication
impl<T: Copy + Mul<Output = T> + PartialEq + Zero> Mul for &Poly<T> {
    type Output = Poly<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Poly<T> {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_length = self.len() + rhs.len() - 1;
        let mut new_coeffs: Vec<T> = vec![T::zero(); new_length];
        for i in 0..self.len() {
            for j in 0..rhs.len() {
                let a = *self.coeffs.get(i).unwrap_or(&T::zero());
                let b = *rhs.coeffs.get(j).unwrap_or(&T::zero());
                let index = i + j;
                new_coeffs[index] = new_coeffs[index] + a * b;
            }
        }
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial multiplication
impl<T: Copy + Mul<Output = T> + PartialEq + Zero> Mul for Poly<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Mul::mul(&self, &rhs)
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Copy + Num> Mul<T> for Poly<T> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        if rhs.is_zero() {
            Self::zero()
        } else {
            for c in &mut self.coeffs {
                *c = *c * rhs;
            }
            self
        }
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Copy + Num> Mul<T> for &Poly<T> {
    type Output = Poly<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut p = self.clone();
        if rhs.is_zero() {
            Self::Output::zero()
        } else {
            for c in &mut p.coeffs {
                *c = *c * rhs;
            }
            p
        }
    }
}

macro_rules! impl_mul_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Mul<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn mul(self, rhs: Poly<Self>) -> Poly<Self> {
                rhs * self
            }
        }
    };
}

impl_mul_for_poly!(
    /// Implementation of f32 and polynomial multiplication
    f32
);
impl_mul_for_poly!(
    /// Implementation of f64 and polynomial multiplication
    f64
);
impl_mul_for_poly!(
    /// Implementation of i8 and polynomial multiplication
    i8
);
impl_mul_for_poly!(
    /// Implementation of u8 and polynomial multiplication
    u8
);
impl_mul_for_poly!(
    /// Implementation of i16 and polynomial multiplication
    i16
);
impl_mul_for_poly!(
    /// Implementation of u16 and polynomial multiplication
    u16
);
impl_mul_for_poly!(
    /// Implementation of i32 and polynomial multiplication
    i32
);
impl_mul_for_poly!(
    /// Implementation of u32 and polynomial multiplication
    u32
);
impl_mul_for_poly!(
    /// Implementation of i64 and polynomial multiplication
    i64
);
impl_mul_for_poly!(
    /// Implementation of u64 and polynomial multiplication
    u64
);
impl_mul_for_poly!(
    /// Implementation of i128 and polynomial multiplication
    i128
);
impl_mul_for_poly!(
    /// Implementation of u128 and polynomial multiplication
    u128
);
impl_mul_for_poly!(
    /// Implementation of isize and polynomial multiplication
    isize
);
impl_mul_for_poly!(
    /// Implementation of usize and polynomial multiplication
    usize
);

/// Implementation of polynomial and float division
impl<T: Copy + Num> Div<T> for Poly<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        let mut result = self.clone();
        for c in &mut result.coeffs {
            *c = *c / rhs;
        }
        result.trim();
        result
    }
}

/// Implementation of polynomial and float division
impl<T: Copy + Num> Div<T> for &Poly<T> {
    type Output = Poly<T>;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self.clone();
        for c in &mut result.coeffs {
            *c = *c / rhs;
        }
        result.trim();
        result
    }
}

impl<T: Float> Div for &Poly<T> {
    type Output = Poly<T>;

    fn div(self, rhs: &Poly<T>) -> Self::Output {
        poly_div_impl(self, rhs).0
    }
}

impl<T: Float> Div for Poly<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs)
    }
}

impl<T: Float> Rem for &Poly<T> {
    type Output = Poly<T>;

    fn rem(self, rhs: &Poly<T>) -> Self::Output {
        poly_div_impl(self, rhs).1
    }
}

impl<T: Float> Rem for Poly<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Rem::rem(&self, &rhs)
    }
}

/// Donald Ervin Knuth, The Art of Computer Programming: Seminumerical algorithms
/// Volume 2, third edition, section 4.6.1
/// Algorithm D: division of polynomials over a field.
#[allow(clippy::many_single_char_names)]
fn poly_div_impl<T: Float>(u: &Poly<T>, v: &Poly<T>) -> (Poly<T>, Poly<T>) {
    let (m, n) = match (u.degree(), v.degree()) {
        (_, None) => panic!("Division by zero polynomial"),
        (None, _) => return (Poly::zero(), Poly::zero()),
        (Some(m), Some(n)) if m < n => return (Poly::zero(), u.clone()),
        (Some(m), Some(n)) => (m, n),
    };

    // 1/v_n
    let vn_rec = v.leading_coeff().recip();

    let mut u_int = u.clone();
    let mut q = Poly {
        coeffs: vec![T::zero(); m - n + 1],
    };

    for k in (0..=m - n).rev() {
        q[k] = u_int[n + k] * vn_rec;
        // n+k-1..=k
        for j in (k..n + k).rev() {
            u_int[j] = u_int[j] - q[k] * v[j - k];
        }
    }

    // (r_n-1, ..., r_0) = (u_n-1, ..., u_0)
    let mut r = Poly::new_from_coeffs(&u_int.coeffs[0..n]);
    r.trim();
    // No need to trim q, its higher degree coefficient is always different from 0.
    (q, r)
}

/// Implementation of the additive identity for polynomials
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// use num_traits::Zero;
/// let zero = Poly::<u8>::zero();
/// assert!(zero.is_zero());
/// ```
impl<T: Copy + Num> Zero for Poly<T> {
    fn zero() -> Self {
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
/// use automatica::polynomial::Poly;
/// use num_traits::One;
/// let one = Poly::<u8>::one();
/// assert!(one.is_one());
/// ```
impl<T: AddAssign + Copy + Num> One for Poly<T> {
    fn one() -> Self {
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
/// let p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// assert_eq!("+1*s +2*s^2 +3*s^3", format!("{}", p));
/// ```
impl<T: Display + One + PartialEq + Signed + Zero> Display for Poly<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        } else if self.len() == 1 {
            return write!(f, "{}", self.coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.coeffs.iter().enumerate() {
            // TODO use approx crate
            //if relative_eq!(*c, 0.0) {
            if *c == T::zero() {
                continue;
            }
            s.push_str(sep);
            #[allow(clippy::float_cmp)] // signum() returns either 1.0 or -1.0
            let sign = if c.signum() == T::one() { "+" } else { "" };
            if i == 0 {
                s.push_str(&format!("{}", c));
            } else if i == 1 {
                s.push_str(&format!("{}{}*s", sign, c));
            } else {
                s.push_str(&format!("{}{}*s^{}", sign, c, i));
            }
            sep = " ";
        }

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn poly_creation_coeffs() {
        let c = [4.3, 5.32];
        assert_eq!(c, Poly::new_from_coeffs(&c).coeffs.as_slice());

        let c2 = [0., 1., 1., 0., 0., 0.];
        assert_eq!([0., 1., 1.], Poly::new_from_coeffs(&c2).coeffs.as_slice());

        let zero: [f64; 1] = [0.];
        assert_eq!(zero, poly!(0., 0.).coeffs.as_slice());

        let int = [1, 2, 3, 4, 5];
        assert_eq!(int, Poly::new_from_coeffs(&int).coeffs.as_slice());

        let float = [0.1_f32, 0.34, 3.43];
        assert_eq!(float, Poly::new_from_coeffs(&float).coeffs.as_slice());
    }

    #[test]
    fn coeffs() {
        let int = [1, 2, 3, 4, 5];
        let p = Poly::new_from_coeffs(&int);
        assert_eq!(int, p.coeffs().as_slice());
    }

    #[test]
    fn poly_creation_roots() {
        assert_eq!(poly!(4., 4., 1.), Poly::new_from_roots(&[-2., -2.]));

        assert_eq!(poly!(4, 4, 1), Poly::new_from_roots(&[-2, -2]));

        assert!(vec![-2., -2.]
            .iter()
            .zip(Poly::new_from_roots(&[-2., -2.]).roots().unwrap().iter())
            .map(|(x, y): (&f64, &f64)| (x - y).abs())
            .all(|x| x < 0.000_001));

        assert!(vec![1.0_f32, 2., 3.]
            .iter()
            .zip(Poly::new_from_roots(&[1., 2., 3.]).roots().unwrap().iter())
            .map(|(x, y): (&f32, &f32)| (x - y).abs())
            .all(|x| x < 0.000_01));

        assert_eq!(
            poly!(0., -2., 1., 1.),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
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
    #[should_panic]
    fn poly_f64_eval_panic() {
        let p = poly!(1.0e200, 2., 3.);
        p.eval(&5.0_f32);
    }

    #[test]
    fn poly_i32_eval() {
        let p = poly!(1.5, 2., 3.);
        assert_eq!(86, p.eval(&5));
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
    fn poly_neg() {
        let p1 = poly!(1., 2.34, -4.2229);
        let p2 = -&p1;
        assert_eq!(p1, -p2);
    }

    #[test]
    fn poly_add() {
        assert_eq!(poly!(4., 4., 4.), poly!(1., 2., 3.) + poly!(3., 2., 1.));

        assert_eq!(poly!(4., 4., 3.), poly!(1., 2., 3.) + poly!(3., 2.));

        assert_eq!(poly!(4., 4., 1.), poly!(1., 2.) + poly!(3., 2., 1.));

        assert_eq!(poly!(4., 4.), poly!(1., 2., 3.) + poly!(3., 2., -3.));

        assert_eq!(poly!(-2., 2., 3.), poly!(1., 2., 3.) + -3.);

        assert_eq!(poly!(0, 2, 3), 2 + poly!(1, 2, 3) + -3);

        assert_eq!(poly!(9.0_f32, 2., 3.), 3. + poly!(1.0_f32, 2., 3.) + 5.);

        let p = poly!(-2, 2, 3);
        let p2 = &p + &p;
        let p3 = &p2 + &p;
        assert_eq!(poly!(-6, 6, 9), p3);
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn poly_sub() {
        assert_eq!(poly!(-2., 0., 2.), poly!(1., 2., 3.) - poly!(3., 2., 1.));

        assert_eq!(poly!(-2., 0., 3.), poly!(1., 2., 3.) - poly!(3., 2.));

        assert_eq!(poly!(-2., 0., -1.), poly!(1., 2.) - poly!(3., 2., 1.));

        assert_eq!(poly!(-2., 0., 6.), poly!(1., 2., 3.) - poly!(3., 2., -3.));

        let p = poly!(1., 1.);
        assert_eq!(Poly::zero(), &p - &p);

        assert_eq!(poly!(-10., 1.), poly!(2., 1.) - 12.);

        assert_eq!(poly!(-1., 1.), 1. - poly!(2., 1.));

        assert_eq!(poly!(-1_i8, 1), 1_i8 - poly!(2, 1));

        assert_eq!(poly!(-10, 1), poly!(2, 1) - 12);
    }

    #[test]
    #[should_panic]
    fn poly_sub_panic() {
        let _ = poly!(1, 2, 3) - 3_u32;
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn poly_mul() {
        assert_eq!(
            poly!(0., 0., -1., 0., -1.),
            poly!(1., 0., 1.) * poly!(0., 0., -1.)
        );

        assert_eq!(Poly::zero(), poly!(1., 0., 1.) * Poly::zero());

        assert_eq!(poly!(1., 0., 1.), poly!(1., 0., 1.) * Poly::one());

        assert_eq!(poly!(-3., 0., -3.), poly!(1., 0., 1.) * poly!(-3.));

        let p = poly!(-3., 0., -3.);
        assert_eq!(poly!(9., 0., 18., 0., 9.), &p * &p);

        assert_eq!(
            poly!(-266.07_f32, 0., -266.07),
            4.9 * poly!(1.0_f32, 0., 1.) * -54.3
        );

        assert_eq!(Poly::zero(), 0. * poly!(1., 0., 1.));

        assert_eq!(Poly::zero(), poly!(1, 0, 1) * 0);

        assert_eq!(Poly::zero(), &poly!(1, 0, 1) * 0);

        assert_eq!(poly!(3, 0, 3), &poly!(1, 0, 1) * 3);
    }

    #[test]
    fn poly_div() {
        assert_eq!(poly!(0.5, 0., 0.5), poly!(1., 0., 1.) / 2.0);

        assert_eq!(poly!(4, 0, 5), poly!(8, 1, 11) / 2);

        let inf = std::f32::INFINITY;
        assert_eq!(Poly::zero(), poly!(1., 0., 1.) / inf);

        assert_eq!(poly!(inf, -inf, inf), poly!(1., -2.3, 1.) / 0.);
    }

    #[test]
    fn poly_mutable_div() {
        let mut p = poly!(3, 4, 5);
        p.div_mut(2);
        assert_eq!(poly!(1, 2, 2), p);
    }

    #[test]
    #[should_panic]
    fn div_panic() {
        let _ = poly_div_impl(&poly!(6., 5., 1.), &poly!(0.));
    }

    #[test]
    fn poly_division_impl() {
        let d1 = poly_div_impl(&poly!(6., 5., 1.), &poly!(2., 1.));
        assert_eq!(poly!(3., 1.), d1.0);
        assert_eq!(poly!(0.), d1.1);

        let d2 = poly_div_impl(&poly!(5., 3., 1.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.5), d2.0);
        assert_eq!(poly!(3.), d2.1);

        let d3 = poly_div_impl(&poly!(3., 1.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.), d3.0);
        assert_eq!(poly!(3., 1.), d3.1);

        let d4 = poly_div_impl(&poly!(0.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.), d4.0);
        assert_eq!(poly!(0.), d4.1);
    }

    #[test]
    fn two_poly_div() {
        let q = poly!(-1., 0., 0., 0., 1.) / poly!(1., 0., 1.);
        assert_eq!(poly!(-1., 0., 1.), q);
    }

    #[test]
    fn two_poly_rem() {
        let r = poly!(-4., 0., -2., 1.) % poly!(-3., 1.);
        assert_eq!(poly!(5.), r);
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

    #[test]
    fn roots() {
        let root1 = -1.;
        let root2 = -2.;
        assert_eq!(Some((root1, root2)), quadratic_roots(3., 2.));

        assert_eq!(None, quadratic_roots(-6., 10.));

        let root3 = 3.;
        assert_eq!(Some((root3, root3)), quadratic_roots(-6., 9.));
    }

    #[test]
    fn poly_roots() {
        let roots = &[0., -1., 1.];
        let p = Poly::new_from_roots(roots);
        assert_eq!(roots, p.roots().unwrap().as_slice());
    }

    #[test]
    fn poly_complx_roots() {
        let p = Poly::new_from_coeffs(&[1.0_f32, 0., 1.]) * poly!(2., 1.);
        assert_eq!(p.complex_roots().len(), 3);
    }

    #[test]
    fn complex_roots() {
        let root1 = Complex::<f64>::new(-1., 0.);
        let root2 = Complex::<f64>::new(-2., 0.);
        assert_eq!((root1, root2), complex_quadratic_roots(3., 2.));

        let root1 = Complex::<f64>::new(-0., -1.);
        let root2 = Complex::<f64>::new(-0., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots(0., 1.));

        let root1 = Complex::<f64>::new(3., -1.);
        let root2 = Complex::<f64>::new(3., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots(-6., 10.));

        let root1 = Complex::<f64>::new(3., 0.);
        assert_eq!((root1, root1), complex_quadratic_roots(-6., 9.));
    }

    #[quickcheck]
    fn leading_coefficient(c: f32) -> bool {
        c == poly!(1., -5., c).leading_coeff()
    }

    #[test]
    fn monic_poly() {
        let p = poly!(-3., 6., 9.);
        let (p2, c) = p.monic();
        assert_eq!(9., c);
        assert_eq!(1., p2.leading_coeff());
    }

    #[test]
    fn monic_mutable_poly() {
        let mut p = poly!(-3., 6., 9.);
        let c = p.monic_mut();
        assert_eq!(9., c);
        assert_eq!(1., p.leading_coeff());
    }
}
