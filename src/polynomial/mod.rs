//! # Polynomials and matrices of polynomials
//!
//! `Poly` implements usual addition, subtraction and multiplication between
//! polynomials, operations with scalars are supported also.
//!
//! Methods for roots finding, companion matrix and evaluation are implemented.
//!
//! `MatrixOfPoly` allows the definition of matrices of polynomials.

use crate::Eval;

use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

use nalgebra::{ComplexField, DMatrix, RealField, Scalar, Schur};
use ndarray::{Array, Array2};
use num_complex::{Complex, Complex64};
use num_traits::{MulAdd, Num, NumCast, One, Signed, Zero};

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// p(x) = c0 + c1*x + c2*x^2 + ...
#[derive(Debug, PartialEq, Clone)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

/// Implementation methods for Poly struct
impl<T> Poly<T> {
    /// Degree of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(2, p.degree());
    /// ```
    pub fn degree(&self) -> usize {
        assert!(
            !self.coeffs.is_empty(),
            "Degree is not defined on empty polynomial"
        );
        self.coeffs.len() - 1
    }
}

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
impl<T: Copy + Zero> Poly<T> {
    /// Extend the polynomial coefficients with 0 to the given degree.
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
        if degree > self.degree() {
            self.coeffs.resize(degree + 1, T::zero());
        }
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Div<Output = T> + One> Poly<T> {
    /// Retrun the monic polynomial and the leading coefficient.
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
        let leading_coeff = *self.coeffs.last().unwrap_or(&T::one());
        let result: Vec<_> = self.coeffs.iter().map(|&x| x / leading_coeff).collect();
        let monic_poly = Self { coeffs: result };

        (monic_poly, leading_coeff)
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
impl<T: Scalar + ComplexField + RealField + Debug> Poly<T> {
    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    /// of the monic polynomial with opposite sign.
    pub(crate) fn companion(&self) -> DMatrix<T> {
        let length = self.degree();
        let hi_coeff = self.coeffs[length];
        DMatrix::from_fn(length, length, |i, j| {
            if j == length - 1 {
                -self.coeffs[i] / hi_coeff // monic polynomial
            } else if i == j + 1 {
                T::one()
            } else {
                T::zero()
            }
        })
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
        // Build the companion matrix
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.eigenvalues().map(|e| e.as_slice().to_vec())
    }

    /// Calculate the complex roots of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let i = num_complex::Complex::i();
    /// assert_eq!(vec![i, -i], p.complex_roots());
    /// ```
    pub fn complex_roots(&self) -> Vec<Complex<T>> {
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.complex_eigenvalues().as_slice().to_vec()
    }
}

/// Implementation methods for Poly struct
impl Poly<f64> {
    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn mul(&self, rhs: &DMatrix<f64>) -> PolyMatrix {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

/// Evaluate the polynomial at the given real or complex number
impl<N, T> Eval<N> for Poly<T>
where
    N: Copy + MulAdd<Output = N> + NumCast + Zero,
    T: Copy + NumCast,
{
    /// Evaluate the polynomial using Horner's method. The evaluation is safe
    /// is the polynomial coefficient can be casted the type `N`.
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
        let new_coeffs = crate::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), Add::add);
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
        let mut result = if self.degree() < rhs.degree() {
            // iterate on rhs and do the subtraction until self has values,
            // then invert the coefficients of rhs
            for i in 0..=rhs.degree() {
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
            crate::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), |x, y| x - y);
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
impl<T: AddAssign + Copy + Num> Mul for Poly<T> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_degree = self.degree() + rhs.degree();
        let mut new_coeffs: Vec<T> = vec![T::zero(); new_degree + 1];
        for i in 0..=self.degree() {
            for j in 0..=rhs.degree() {
                let a = *self.coeffs.get(i).unwrap_or(&T::zero());
                let b = *rhs.coeffs.get(j).unwrap_or(&T::zero());
                new_coeffs[i + j] += a * b;
            }
        }
        Self::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Copy + Num> Mul<T> for Poly<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        if rhs.is_zero() {
            Self::zero()
        } else {
            let mut result = self.clone();
            for c in &mut result.coeffs {
                *c = *c * rhs;
            }
            result
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

/// Implementation of the additive identity for polynomials
impl<T: Copy + Num + Zero> Zero for Poly<T> {
    fn zero() -> Self {
        Self {
            coeffs: vec![T::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs == vec![T::zero()]
    }
}

/// Implementation of the multiplicative identity for polynomials
impl<T: AddAssign + Copy + Num> One for Poly<T> {
    fn one() -> Self {
        Self {
            coeffs: vec![T::one()],
        }
    }

    fn is_one(&self) -> bool {
        self.coeffs == vec![T::one()]
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
        } else if self.degree() == 0 {
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
        assert_eq!(zero, Poly::new_from_coeffs(&[0., 0.]).coeffs.as_slice());

        let int = [1, 2, 3, 4, 5];
        assert_eq!(int, Poly::new_from_coeffs(&int).coeffs.as_slice());

        let float = [0.1_f32, 0.34, 3.43];
        assert_eq!(float, Poly::new_from_coeffs(&float).coeffs.as_slice());
    }

    #[test]
    fn poly_creation_roots() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_roots(&[-2., -2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4, 4, 1]),
            Poly::new_from_roots(&[-2, -2])
        );

        assert!(vec![-2., -2.]
            .iter()
            .zip(Poly::new_from_roots(&[-2., -2.]).roots().unwrap().iter())
            .map(|(x, y): (&f64, &f64)| (x - y).abs())
            .all(|x| x < 0.000001));

        assert!(vec![1.0_f32, 2., 3.]
            .iter()
            .zip(Poly::new_from_roots(&[1., 2., 3.]).roots().unwrap().iter())
            .map(|(x, y): (&f32, &f32)| (x - y).abs())
            .all(|x| x < 0.00001));

        assert_eq!(
            Poly::new_from_coeffs(&[0., -2., 1., 1.]),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
    }

    #[test]
    fn poly_eval() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(86., p.eval(&5.));

        assert_eq!(0.0, Poly::<f64>::zero().eval(&6.4));

        let p2 = Poly::new_from_coeffs(&[3, 4, 1]);
        assert_eq!(143, p2.eval(&10));
    }

    #[test]
    #[should_panic]
    fn poly_f64_eval_panic() {
        let p = Poly::new_from_coeffs(&[1.0e200, 2., 3.]);
        p.eval(&5.0_f32);
    }

    #[test]
    fn poly_i32_eval() {
        let p = Poly::new_from_coeffs(&[1.5, 2., 3.]);
        assert_eq!(86, p.eval(&5));
    }

    #[test]
    fn poly_cmplx_eval() {
        let p = Poly::new_from_coeffs(&[1., 1., 1.]);
        let c = Complex::new(1.0, 1.0);
        let res = Complex::new(2.0, 3.0);
        assert_eq!(res, p.eval(&c));

        assert_eq!(
            Complex::zero(),
            Poly::<f64>::new_from_coeffs(&[]).eval(&Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_add() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) + Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_coeffs(&[1., 2.]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) + Poly::new_from_coeffs(&[3., 2., -3.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 2., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) + -3.
        );

        assert_eq!(
            Poly::new_from_coeffs(&[0, 2, 3]),
            2 + Poly::new_from_coeffs(&[1, 2, 3]) + -3
        );

        assert_eq!(
            Poly::new_from_coeffs(&[9.0_f32, 2., 3.]),
            3. + Poly::new_from_coeffs(&[1.0_f32, 2., 3.]) + 5.
        );

        let p = Poly::new_from_coeffs(&[-2, 2, 3]);
        let p2 = &p + &p;
        let p3 = &p2 + &p;
        assert_eq!(Poly::new_from_coeffs(&[-6, 6, 9]), p3);
    }

    #[test]
    fn poly_sub() {
        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 2.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) - Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., -1.]),
            Poly::new_from_coeffs(&[1., 2.]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 6.]),
            Poly::new_from_coeffs(&[1., 2., 3.]) - Poly::new_from_coeffs(&[3., 2., -3.])
        );

        let p = Poly::new_from_coeffs(&[1., 1.]);
        assert_eq!(Poly::zero(), &p - &p);

        assert_eq!(
            Poly::new_from_coeffs(&[-10., 1.]),
            Poly::new_from_coeffs(&[2., 1.]) - 12.
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-1., 1.]),
            1. - Poly::new_from_coeffs(&[2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-1i8, 1]),
            1i8 - Poly::new_from_coeffs(&[2, 1])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-10, 1]),
            Poly::new_from_coeffs(&[2, 1]) - 12
        );
    }

    #[test]
    #[should_panic]
    fn poly_sub_panic() {
        let _ = Poly::new_from_coeffs(&[1, 2, 3]) - 3u32;
    }

    #[test]
    fn poly_mul() {
        assert_eq!(
            Poly::new_from_coeffs(&[0., 0., -1., 0., -1.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[0., 0., -1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[0.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[0.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[1., 0., 1.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-3., 0., -3.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[-3.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-266.07_f32, 0., -266.07]),
            4.9 * Poly::new_from_coeffs(&[1.0_f32, 0., 1.]) * -54.3
        );

        assert_eq!(Poly::zero(), 0. * Poly::new_from_coeffs(&[1., 0., 1.]));

        assert_eq!(Poly::zero(), Poly::new_from_coeffs(&[1, 0, 1]) * 0);
    }

    #[test]
    fn poly_div() {
        assert_eq!(
            Poly::new_from_coeffs(&[0.5, 0., 0.5]),
            Poly::new_from_coeffs(&[1., 0., 1.]) / 2.0
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4, 0, 5]),
            Poly::new_from_coeffs(&[8, 1, 11]) / 2
        );

        let inf = std::f32::INFINITY;
        assert_eq!(Poly::zero(), Poly::new_from_coeffs(&[1., 0., 1.]) / inf);

        assert_eq!(
            Poly::new_from_coeffs(&[inf, -inf, inf]),
            Poly::new_from_coeffs(&[1., -2.3, 1.]) / 0.
        );
    }

    #[test]
    fn indexing() {
        assert_eq!(3., Poly::new_from_coeffs(&[1., 3.])[1]);

        let mut p = Poly::new_from_roots(&[1., 4., 5.]);
        p[2] = 3.;
        assert_eq!(Poly::new_from_coeffs(&[-20., 29., 3., 1.]), p);
    }

    #[test]
    fn identities() {
        assert!(Poly::<f64>::zero().is_zero());
        assert!(Poly::<f64>::one().is_one());

        assert!(Poly::<f32>::zero().is_zero());
        assert!(Poly::<f32>::one().is_one());

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
}

/// Polynomial matrix object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// P(x) = C0 + C1*x + C2*x^2 + ...
#[derive(Clone, Debug)]
pub(crate) struct PolyMatrix {
    pub(crate) matr_coeffs: Vec<DMatrix<f64>>,
}

/// Implementation methods for `PolyMatrix` struct
impl PolyMatrix {
    /// Create a new polynomial matrix given a slice of matrix coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of matrix coefficients
    pub(crate) fn new_from_coeffs(matr_coeffs: &[DMatrix<f64>]) -> Self {
        let shape = matr_coeffs[0].shape();
        assert!(matr_coeffs.iter().all(|c| c.shape() == shape));
        let mut pm = Self {
            matr_coeffs: matr_coeffs.into(),
        };
        pm.trim();
        debug_assert!(!pm.matr_coeffs.is_empty());
        pm
    }

    /// Degree of the polynomial matrix
    pub(crate) fn degree(&self) -> usize {
        assert!(
            !self.matr_coeffs.is_empty(),
            "Degree is not defined on empty polynomial matrix"
        );
        self.matr_coeffs.len() - 1
    }

    /// Implementation of polynomial matrix and matrix multiplication
    ///
    /// PolyMatrix * DMatrix
    pub(crate) fn right_mul(&self, rhs: &DMatrix<f64>) -> Self {
        let result: Vec<_> = self.matr_coeffs.iter().map(|x| x * rhs).collect();
        Self::new_from_coeffs(&result)
    }

    /// Implementation of matrix and polynomial matrix multiplication
    ///
    /// DMatrix * PolyMatrix
    pub(crate) fn left_mul(&self, lhs: &DMatrix<f64>) -> Self {
        let res: Vec<_> = self.matr_coeffs.iter().map(|r| lhs * r).collect();
        Self::new_from_coeffs(&res)
    }

    /// Trim the zeros coefficients of high degree terms
    fn trim(&mut self) {
        let rows = self.matr_coeffs[0].nrows();
        let cols = self.matr_coeffs[0].ncols();
        let zero = DMatrix::zeros(rows, cols);
        if let Some(p) = self.matr_coeffs.iter().rposition(|c| c != &zero) {
            self.matr_coeffs.truncate(p + 1);
        } else {
            self.matr_coeffs.resize(1, zero);
        }
    }
}

impl Eval<DMatrix<Complex64>> for PolyMatrix {
    fn eval(&self, s: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        // transform matr_coeffs in complex numbers matrices
        //
        // ┌     ┐ ┌       ┐ ┌       ┐ ┌     ┐
        // │P1 P2│=│a01 a02│+│a11 a12│*│s1 s2│
        // │P3 P4│ │a03 a04│ │a13 a14│ │s1 s2│
        // └     ┘ └       ┘ └       ┘ └     ┘
        // `*` is the element by element multiplication
        // If i have a 2x2 matr_coeff the result shall be a 2x2 matrix,
        // because otherwise i will sum P1 and P2 (P3 and P4)
        let rows = self.matr_coeffs[0].nrows();
        let cols = self.matr_coeffs[0].ncols();

        let mut result = DMatrix::from_element(rows, cols, Complex64::zero());

        for mc in self.matr_coeffs.iter().rev() {
            let mcplx = mc.map(|x| Complex64::new(x, 0.0));
            result = result.component_mul(s) + mcplx;
        }
        result
    }
}

/// Implementation of polynomial matrices addition
impl Add<PolyMatrix> for PolyMatrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Check which polynomial matrix has the highest degree
        let new_coeffs = if self.degree() < rhs.degree() {
            let mut result = rhs.matr_coeffs.to_vec();
            for (i, c) in self.matr_coeffs.iter().enumerate() {
                result[i] += c;
            }
            result
        } else if rhs.degree() < self.degree() {
            let mut result = self.matr_coeffs.to_owned();
            for (i, c) in rhs.matr_coeffs.iter().enumerate() {
                result[i] += c;
            }
            result
        } else {
            crate::zip_with(&self.matr_coeffs, &rhs.matr_coeffs, |l, r| l + r)
        };
        Self::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of read only indexing of polynomial matrix
/// returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl Index<usize> for PolyMatrix {
    type Output = DMatrix<f64>;

    fn index(&self, i: usize) -> &DMatrix<f64> {
        &self.matr_coeffs[i]
    }
}

/// Implementation of polynomial matrix printing
impl Display for PolyMatrix {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.degree() == 0 {
            return write!(f, "{}", self.matr_coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.matr_coeffs.iter().enumerate() {
            if c.iter().all(|&x| relative_eq!(x, 0.0)) {
                continue;
            }
            s.push_str(sep);
            if i == 0 {
                s.push_str(&format!("{}", c));
            } else if i == 1 {
                s.push_str(&format!("+{}*s", c));
            } else {
                s.push_str(&format!("+{}*s^{}", c, i));
            }
            sep = " ";
        }

        write!(f, "{}", s)
    }
}

/// Polynomial matrix object
///
/// Contains the matrix of polynomials
///
/// P(x) = [[P1, P2], [P3, P4]]
#[derive(Debug)]
pub struct MatrixOfPoly {
    pub(crate) matrix: Array2<Poly<f64>>,
}

/// Implementation methods for MP struct
impl MatrixOfPoly {
    /// Create a new polynomial matrix given a vector of polynomials.
    ///
    /// # Arguments
    ///
    /// * `rows` - number of rows of the matrix
    /// * `cols` - number of columns of the matrix
    /// * `data` - vector of polynomials in row major order
    ///
    /// # Panics
    ///
    /// Panics if the matrix cannot be build from given arguments.
    fn new(rows: usize, cols: usize, data: Vec<Poly<f64>>) -> Self {
        Self {
            matrix: Array::from_shape_vec((rows, cols), data)
                .expect("Input data do not allow to create the matrix"),
        }
    }

    /// Extract the transfer function from the matrix if is the only one.
    /// Use to get Single Input Single Output transfer function.
    pub fn siso(&self) -> Option<&Poly<f64>> {
        if self.matrix.shape() == [1, 1] {
            self.matrix.first()
        } else {
            None
        }
    }
}

/// Implement conversion between different representations.
impl From<PolyMatrix> for MatrixOfPoly {
    fn from(pm: PolyMatrix) -> Self {
        let coeffs = pm.matr_coeffs; // vector of matrices
        let rows = coeffs[0].nrows();
        let cols = coeffs[0].ncols();

        // Each vector contains the corresponding matrix in coeffs,
        // so each vector contains the coefficients of the polynomial
        // with the same order (increasing).
        let vectorized_coeffs: Vec<Vec<_>> = coeffs
            .iter()
            .map(|c| c.transpose().as_slice().to_vec())
            .collect();

        // Crate a vector containing the vector of coefficients a single
        // polynomial in row major mode with respect to the initial
        // vector of matrices.
        let mut tmp: Vec<Vec<f64>> = vec![vec![]; rows * cols];
        for order in vectorized_coeffs {
            for (i, value) in order.into_iter().enumerate() {
                tmp[i].push(value);
            }
        }

        let polys: Vec<Poly<f64>> = tmp.iter().map(|p| Poly::new_from_coeffs(&p)).collect();
        Self::new(rows, cols, polys)
    }
}

/// Implementation of matrix of polynomials printing
impl Display for MatrixOfPoly {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn mp_creation() {
        let c = [4.3, 5.32];
        let p = Poly::new_from_coeffs(&c);
        let v = vec![p.clone(), p.clone(), p.clone(), p.clone()];
        let mp = MatrixOfPoly::new(2, 2, v);
        let expected = "[[4.3 +5.32*s, 4.3 +5.32*s],\n [4.3 +5.32*s, 4.3 +5.32*s]]";
        assert_eq!(expected, format!("{}", &mp));
    }
}
