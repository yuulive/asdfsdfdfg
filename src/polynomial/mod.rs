//! # Polynomials and matrices of polynomials
//!
//! `Poly` implements usual addition, subtraction and multiplication between
//! polynomials, operations with scalars are supported also.
//!
//! Methods for roots finding, companion matrix and evaluation are implemented.
//!
//! At the moment coefficients are of type `f64`.
//!
//! `MatrixOfPoly` allows the definition of matrices of polynomials.

use crate::Eval;

use std::fmt;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub};

use nalgebra::{DMatrix, Schur};
use ndarray::{Array, Array2};
use num_complex::Complex64;
use num_traits::{Float, FromPrimitive, MulAdd, One, Zero};

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// p(x) = c0 + c1*x + c2*x^2 + ...
#[derive(Debug, PartialEq, Clone)]
pub struct Poly<F: Float> {
    coeffs: Vec<F>,
}

/// Implementation methods for Poly struct
impl<F: Float> Poly<F> {
    /// Create a new polynomial given a slice of real coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    pub fn new_from_coeffs(coeffs: &[F]) -> Self {
        let mut p = Self {
            coeffs: coeffs.into(),
        };
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }
    /// Trim the zeros coefficients of high degree terms.
    /// It uses f64::EPSILON as both absolute and relative difference for
    /// zero equality check.
    fn trim(&mut self) {
        self.trim_complete(F::epsilon(), F::epsilon());
    }

    /// Trim the zeros coefficients of high degree terms
    ///
    /// # Arguments
    /// * `epsilon` - absolute difference for zero equality check
    /// * `max_relative` - maximum relative difference for zero equality check
    fn trim_complete(&mut self, _epsilon: F, _max_relative: F) {
        // TODO try to use assert macro.
        //.rposition(|&c| relative_ne!(c, 0.0, epsilon = epsilon, max_relative = max_relative))
        if let Some(p) = self.coeffs.iter().rposition(|&c| c != F::zero()) {
            let new_length = p + 1;
            self.coeffs.truncate(new_length);
        } else {
            self.coeffs.resize(1, F::zero());
        }
    }

    /// Degree of the polynomial
    pub fn degree(&self) -> usize {
        assert!(
            !self.coeffs.is_empty(),
            "Degree is not defined on empty polynomial"
        );
        self.coeffs.len() - 1
    }

    /// Vector of the polynomial's coefficients
    pub fn coeffs(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    /// Extend the polynomial coefficients with 0 to the given degree.
    /// It does not truncate the polynomial.
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the new highest coefficient.
    pub fn extend(&mut self, degree: usize) {
        if degree > self.degree() {
            self.coeffs.resize(degree + 1, F::zero());
        }
    }

    /// Retrun the monic polynomial and the leading coefficient.
    pub fn monic(&self) -> (Self, F) {
        let leading_coeff = *self.coeffs.last().unwrap_or(&F::one());
        let result: Vec<_> = self.coeffs.iter().map(|&x| x / leading_coeff).collect();
        let monic_poly = Self { coeffs: result };

        (monic_poly, leading_coeff)
    }
}

/// Implementation methods for Poly struct
impl Poly<f64> {
    /// Create a new polynomial given a slice of real roots
    ///
    /// # Arguments
    ///
    /// * `roots` - slice of roots
    pub fn new_from_roots(roots: &[f64]) -> Self {
        let mut p = roots.iter().fold(Self { coeffs: vec![1.] }, |acc, &r| {
            acc * Self {
                coeffs: vec![-r, 1.],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    /// of the monic polynomial with opposite sign.
    pub(crate) fn companion(&self) -> DMatrix<f64> {
        let length = self.degree();
        let hi_coeff = self.coeffs[length];
        DMatrix::from_fn(length, length, |i, j| {
            if j == length - 1 {
                -self.coeffs[i] / hi_coeff // monic polynomial
            } else if i == j + 1 {
                1.
            } else {
                0.
            }
        })
    }

    /// Calculate the real roots of the polynomial
    pub fn roots(&self) -> Option<Vec<f64>> {
        // Build the companion matrix
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.eigenvalues().map(|e| e.as_slice().to_vec())
    }

    /// Calculate the complex roots of the polynomial
    pub fn complex_roots(&self) -> Vec<Complex64> {
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.complex_eigenvalues().as_slice().to_vec()
    }

    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn mul(&self, rhs: &DMatrix<f64>) -> PolyMatrix {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

/// Evaluate the polynomial at the given real or complex number
impl<N> Eval<N> for Poly<f64>
where
    N: Copy + FromPrimitive + MulAdd<Output = N> + Zero,
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
    /// The method panics if the conversion from f64 to type `N` fails.
    fn eval(&self, x: &N) -> N {
        self.coeffs.iter().rev().fold(N::zero(), |acc, &c| {
            acc.mul_add(*x, N::from_f64(c).unwrap())
        })
    }
}

/// Implement read only indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<F: Float> Index<usize> for Poly<F> {
    type Output = F;

    fn index(&self, i: usize) -> &F {
        &self.coeffs[i]
    }
}

/// Implement mutable indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<F: Float> IndexMut<usize> for Poly<F> {
    fn index_mut(&mut self, i: usize) -> &mut F {
        &mut self.coeffs[i]
    }
}

/// Implementation of polynomial addition
impl<F: Float> Add<Poly<F>> for Poly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Check which polynomial has the highest degree
        let new_coeffs = if self.degree() < rhs.degree() {
            let mut result = rhs.coeffs.to_vec();
            for (i, c) in self.coeffs.iter().enumerate() {
                result[i] = result[i] + *c;
            }
            result
        } else if rhs.degree() < self.degree() {
            let mut result = self.coeffs.to_owned();
            for (i, c) in rhs.coeffs.iter().enumerate() {
                result[i] = result[i] + *c;
            }
            result
        } else {
            crate::zip_with(&self.coeffs, &rhs.coeffs, |&l, &r| l + r)
        };
        Self::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float addition
impl<F: Float> Add<F> for Poly<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self {
        let mut result = self.clone();
        result[0] = result[0] + rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result
    }
}

/// Implementation of f64 and polynomial addition
impl Add<Poly<f64>> for f64 {
    type Output = Poly<f64>;

    fn add(self, rhs: Poly<f64>) -> Poly<f64> {
        rhs + self
    }
}

/// Implementation of f32 and polynomial addition
impl Add<Poly<f32>> for f32 {
    type Output = Poly<f32>;

    fn add(self, rhs: Poly<f32>) -> Poly<f32> {
        rhs + self
    }
}

/// Implementation of polynomial subtraction
impl<F: Float> Sub for Poly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Just multiply 'rhs' by -1 and use addition.
        let sub_p: Vec<_> = rhs.coeffs.iter().map(|&c| -c).collect();
        self.add(Self::new_from_coeffs(&sub_p))
    }
}

/// Implementation of polynomial and float subtraction
impl<F: Float> Sub<F> for Poly<F> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self {
        let mut result = self.clone();
        result[0] = result[0] - rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result
    }
}

/// Implementation of f64 and polynomial subtraction
impl Sub<Poly<f64>> for f64 {
    type Output = Poly<f64>;

    fn sub(self, rhs: Poly<f64>) -> Poly<f64> {
        let mut result = rhs.clone();
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result[0] = self - result[0];
        result
    }
}

/// Implementation of f32 and polynomial subtraction
impl Sub<Poly<f32>> for f32 {
    type Output = Poly<f32>;

    fn sub(self, rhs: Poly<f32>) -> Poly<f32> {
        let mut result = rhs.clone();
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        result[0] = self - result[0];
        result
    }
}

/// Implementation of polynomial multiplication
impl<F: Float + AddAssign> Mul for Poly<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_degree = self.degree() + rhs.degree();
        let mut new_coeffs: Vec<F> = vec![F::zero(); new_degree + 1];
        for i in 0..=self.degree() {
            for j in 0..=rhs.degree() {
                let a = *self.coeffs.get(i).unwrap_or(&F::zero());
                let b = *rhs.coeffs.get(j).unwrap_or(&F::zero());
                new_coeffs[i + j] += a * b;
            }
        }
        Self::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float multiplication
impl<F: Float> Mul<F> for Poly<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        let result: Vec<_> = self.coeffs.iter().map(|&x| x * rhs).collect();
        Self::new_from_coeffs(&result)
    }
}

/// Implementation of f64 and polynomial multiplication
impl Mul<Poly<f64>> for f64 {
    type Output = Poly<f64>;

    fn mul(self, rhs: Poly<f64>) -> Poly<f64> {
        rhs * self
    }
}

/// Implementation of f32 and polynomial multiplication
impl Mul<Poly<f32>> for f32 {
    type Output = Poly<f32>;

    fn mul(self, rhs: Poly<f32>) -> Poly<f32> {
        rhs * self
    }
}

/// Implementation of polynomial and float division
impl<F: Float> Div<F> for Poly<F> {
    type Output = Self;

    fn div(self, rhs: F) -> Self {
        let result: Vec<_> = self.coeffs.iter().map(|&x| x / rhs).collect();
        Self::new_from_coeffs(&result)
    }
}

/// Implementation of the additive identity for polynomials
impl<F: Float> Zero for Poly<F> {
    fn zero() -> Self {
        Self {
            coeffs: vec![F::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs == vec![F::zero()]
    }
}

/// Implementation of the multiplicative identity for polynomials
impl<F: Float + AddAssign> One for Poly<F> {
    fn one() -> Self {
        Self {
            coeffs: vec![F::one()],
        }
    }

    fn is_one(&self) -> bool {
        self.coeffs == vec![F::one()]
    }
}

/// Implement printing of polynomial
impl fmt::Display for Poly<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        } else if self.degree() == 0 {
            return write!(f, "{}", self.coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.coeffs.iter().enumerate() {
            if relative_eq!(*c, 0.0) {
                continue;
            }
            s.push_str(sep);
            #[allow(clippy::float_cmp)] // signum() returns either 1.0 or -1.0
            let sign = if c.signum() == 1.0 { "+" } else { "" };
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
    }

    #[test]
    fn poly_creation_roots() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_roots(&[-2., -2.])
        );

        assert!(vec![-2., -2.]
            .iter()
            .zip(Poly::new_from_roots(&[-2., -2.]).roots().unwrap())
            .map(|(x, y)| (x - y).abs())
            .all(|x| x < 0.000001));

        assert!(vec![1., 2., 3.]
            .iter()
            .zip(Poly::new_from_roots(&[1., 2., 3.]).roots().unwrap())
            .map(|(x, y)| (x - y).abs())
            .all(|x| x < 0.000001));

        assert_eq!(
            Poly::new_from_coeffs(&[0., -2., 1., 1.]),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
    }

    #[test]
    fn poly_f64_eval() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(86., p.eval(&5.));

        assert_eq!(0.0, Poly::new_from_coeffs(&[]).eval(&6.4));
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
            Poly::new_from_coeffs(&[]).eval(&Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_add() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_coeffs(&[1., 2.,]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2., -3.])
        );
    }

    #[test]
    fn poly_sub() {
        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 2.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., -1.]),
            Poly::new_from_coeffs(&[1., 2.,]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 6.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2., -3.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[]),
            Poly::new_from_coeffs(&[1., 1.]) - Poly::new_from_coeffs(&[1., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-10., 1.]),
            Poly::new_from_coeffs(&[2., 1.]) - 12.
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-1., 1.]),
            1. - Poly::new_from_coeffs(&[2., 1.])
        );
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
impl fmt::Display for PolyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
impl fmt::Display for MatrixOfPoly {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
