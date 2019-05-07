use crate::Eval;

use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use nalgebra::{DMatrix, DVector, Schur};
use num_complex::{Complex, Complex64};
use num_traits::{One, Zero};

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// p(x) = c0 + c1*x + c2*x^2 + ...
#[derive(Debug, PartialEq, Clone)]
pub struct Poly {
    coeffs: Vec<f64>,
}

/// Implementation methods for Poly struct
impl Poly {
    /// Create a new polynomial given a slice of real coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    pub fn new_from_coeffs(coeffs: &[f64]) -> Self {
        let mut p = Poly {
            coeffs: coeffs.into(),
        };
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Create a new polynomial given a slice of real roots
    ///
    /// # Arguments
    ///
    /// * `roots` - slice of roots
    pub fn new_from_roots(roots: &[f64]) -> Self {
        let mut p = roots.iter().fold(Poly { coeffs: vec![1.] }, |acc, &r| {
            acc * Poly {
                coeffs: vec![-r, 1.],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Trim the zeros coefficients of high degree terms
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    fn trim(&mut self) {
        if let Some(p) = self.coeffs.iter().rposition(|&c| c != 0.0) {
            self.coeffs.truncate(p + 1);
        } else {
            self.coeffs.resize(1, 0.0);
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
    pub fn coeffs(&self) -> Vec<f64> {
        self.coeffs.clone()
    }

    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    pub fn companion(&self) -> DMatrix<f64> {
        let length = self.degree();
        let hi_coeff = self.coeffs[length];
        DMatrix::from_fn(length, length, |i, j| {
            if j == length - 1 {
                -self.coeffs[i] / hi_coeff
            } else if i == j + 1 {
                1.
            } else {
                0.
            }
        })
    }

    /// Calculate the real roots of the polynomial
    pub fn roots(&self) -> Option<DVector<f64>> {
        // Build the companion matrix
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.eigenvalues()
    }

    /// Calculate the complex roots of the polynomial
    pub fn complex_roots(&self) -> DVector<Complex64> {
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.complex_eigenvalues()
    }
}

/// Evaluate the polynomial at the given float number
impl Eval<Complex64> for Poly {
    fn eval(&self, x: Complex64) -> Complex64 {
        self.coeffs
            .iter()
            .rev()
            .fold(Complex::zero(), |acc, &c| acc * x + c)
    }
}

/// Evaluate the polynomial at the given complex number
impl Eval<f64> for Poly {
    fn eval(&self, x: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }
}

/// Implement read only indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl Index<usize> for Poly {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.coeffs[i]
    }
}

/// Implement mutable indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl IndexMut<usize> for Poly {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.coeffs[i]
    }
}

/// Implementation of polynomial addition
impl Add<Poly> for Poly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Check which polynomial has the highest degree
        let new_coeffs = if self.degree() < rhs.degree() {
            let mut res = rhs.coeffs.to_vec();
            for (i, c) in self.coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else if rhs.degree() < self.degree() {
            let mut res = self.coeffs.to_owned();
            for (i, c) in rhs.coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else {
            zip_with(&self.coeffs, &rhs.coeffs, |l, r| l + r)
        };
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float addition
impl Add<f64> for Poly {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        let mut res = self.coeffs.to_owned();
        res[0] += rhs;
        Poly::new_from_coeffs(&res)
    }
}

/// Implementation of float and polynomial addition
impl Add<Poly> for f64 {
    type Output = Poly;

    fn add(self, rhs: Poly) -> Poly {
        rhs + self
    }
}

/// Implementation of polynomial subtraction
impl Sub for Poly {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // Just multiply 'rhs' by -1 and use addition.
        let sub_p: Vec<_> = rhs.coeffs.iter().map(|&c| -c).collect();
        self.add(Poly::new_from_coeffs(&sub_p))
    }
}

/// Implementation of polynomial and float subtraction
impl Sub<f64> for Poly {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        let mut res = self.coeffs.to_owned();
        res[0] -= rhs;
        Poly::new_from_coeffs(&res)
    }
}

/// Implementation of float and polynomial subtraction
impl Sub<Poly> for f64 {
    type Output = Poly;

    fn sub(self, rhs: Poly) -> Poly {
        let mut res = rhs.coeffs.to_owned();
        res[0] -= self;
        Poly::new_from_coeffs(&res)
    }
}

/// Implementation of polynomial multiplication
impl Mul for Poly {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_degree = self.degree() + rhs.degree();
        let mut new_coeffs: Vec<f64> = vec![0.; new_degree + 1];
        for i in 0..=self.degree() {
            for j in 0..=rhs.degree() {
                let a = self.coeffs.get(i).unwrap_or(&0.);
                let b = rhs.coeffs.get(j).unwrap_or(&0.);
                new_coeffs[i + j] += a * b;
            }
        }
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and float multiplication
impl Mul<f64> for Poly {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        let res: Vec<_> = self.coeffs.iter().map(|x| x * rhs).collect();
        Poly::new_from_coeffs(&res)
    }
}

/// Implementation of float and polynomial multiplication
impl Mul<Poly> for f64 {
    type Output = Poly;

    fn mul(self, rhs: Poly) -> Poly {
        rhs * self
    }
}

impl Mul<DMatrix<f64>> for Poly {
    type Output = PolyMatrix;

    fn mul(self, rhs: DMatrix<f64>) -> PolyMatrix {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let res: Vec<_> = self.coeffs.iter().map(|&c| c * &rhs).collect();
        PolyMatrix::new_from_coeffs(&res)
    }
}

/// Implementation of polynomial and float division
impl Div<f64> for Poly {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        let res: Vec<_> = self.coeffs.iter().map(|x| x / rhs).collect();
        Poly::new_from_coeffs(&res)
    }
}

/// Implementation of the additive identity for polynomials
impl Zero for Poly {
    fn zero() -> Self {
        Poly { coeffs: vec![0.0] }
    }

    fn is_zero(&self) -> bool {
        self.coeffs == vec![0.0]
    }
}

/// Implementation of the multiplicative identity for polynomials
impl One for Poly {
    fn one() -> Self {
        Poly { coeffs: vec![1.0] }
    }

    fn is_one(&self) -> bool {
        self.coeffs == vec![1.0]
    }
}

/// Implement printing of polynomial
impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        } else if self.degree() == 0 {
            return write!(f, "{}", self.coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.coeffs.iter().enumerate() {
            if *c == 0.0 {
                continue;
            }
            s.push_str(sep);
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

// fn zipWith<U, C>(combo: C, left: U, right: U) -> impl Iterator
// where
//     U: Iterator,
//     C: FnMut(U::Item, U::Item) -> U::Item,
// {
//     left.zip(right).map(move |(l, r)| combo(l, r))
// }
/// Zip two slices with the given function
///
/// # Arguments
///
/// * `left` - first slice to zip
/// * `right` - second slice to zip
/// * `f` - function used to zip the two lists
fn zip_with<T, F>(left: &[T], right: &[T], mut f: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> T,
{
    left.iter().zip(right).map(|(l, r)| f(l, r)).collect()
}

/// Evaluate rational function at x
///
/// # Arguments
///
/// * `x` - Value for the evaluation
/// * `num` - Coefficients of the numerator polynomial. First element is the higher order coefficient
/// * `denom` - Coefficients of the denominator polynomial. First element is the higher order coefficient
// pub fn ratevl<T>(x: T, num: &[T], denom: &[T]) -> T
// where
//     T: Float,
// {
//     if x <= T::one() {
//         polynom_eval(x, num) / polynom_eval(x, denom)
//     } else {
//         // To avoid overflow the result is the same if coefficients are
//         // reversed and evaluated at 1/x
//         let z = x.recip();
//         polynom_eval_rev(z, num) / polynom_eval_rev(z, denom)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn polynom_ratevl_test() {
    //     let num = [1.0, 4.0, 3.0];
    //     let den = [2.0, -6.5, 0.4];
    //     let ratio1 = ratevl(3.3, &num, &den);
    //     assert_eq!(ratio1, 37.10958904109594);

    //     let num2 = [3.0, 4.0, 1.0];
    //     let den2 = [0.4, -6.5, 2.0];
    //     let ratio2 = ratevl(1.0 / 3.3, &num2, &den2);
    //     assert_eq!(ratio2, 37.10958904109594);

    //     assert_eq!(ratio1, ratio2);
    // }

    #[test]
    fn poly_creation_coeffs_test() {
        let c = [4.3, 5.32];
        assert_eq!(c, Poly::new_from_coeffs(&c).coeffs.as_slice());

        let c2 = [0., 1., 1., 0., 0., 0.];
        assert_eq!([0., 1., 1.], Poly::new_from_coeffs(&c2).coeffs.as_slice());

        let zero: [f64; 1] = [0.];
        assert_eq!(zero, Poly::new_from_coeffs(&[0., 0.]).coeffs.as_slice());
    }

    #[test]
    fn poly_creation_roots_test() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_roots(&[-2., -2.])
        );

        assert!((DVector::from_element(2, -2.)
            - Poly::new_from_roots(&[-2., -2.]).roots().unwrap())
        .abs()
        .iter()
        .all(|&x| x < 0.000001));

        assert!((DVector::from_column_slice(&[1., 2., 3.])
            - Poly::new_from_roots(&[1., 2., 3.]).roots().unwrap())
        .abs()
        .iter()
        .all(|&x| x < 0.000001));

        assert_eq!(
            Poly::new_from_coeffs(&[0., -2., 1., 1.]),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
    }

    #[test]
    fn poly_f64_eval_test() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(86., p.eval(5.));

        assert_eq!(0.0, Poly::new_from_coeffs(&[]).eval(6.4));
    }

    #[test]
    fn poly_cmplx_eval_test() {
        let p = Poly::new_from_coeffs(&[1., 1., 1.]);
        let c = Complex::new(1.0, 1.0);
        let res = Complex::new(2.0, 3.0);
        assert_eq!(res, p.eval(c));

        assert_eq!(
            Complex::zero(),
            Poly::new_from_coeffs(&[]).eval(Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_add_test() {
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
    fn poly_sub_test() {
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
    }

    #[test]
    fn poly_mul_test() {
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
    fn indexing_test() {
        assert_eq!(3., Poly::new_from_coeffs(&[1., 3.])[1]);

        let mut p = Poly::new_from_roots(&[1., 4., 5.]);
        p[2] = 3.;
        assert_eq!(Poly::new_from_coeffs(&[-20., 29., 3., 1.]), p);
    }

    #[test]
    fn identities_test() {
        assert!(Poly::zero().is_zero());
        assert!(Poly::one().is_one());
    }
}

/// Polynomial matrix object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// P(x) = C0 + C1*x + C2*x^2 + ...
#[derive(Clone, Debug)]
pub struct PolyMatrix {
    matr_coeffs: Vec<DMatrix<f64>>,
}

/// Implementation methods for PolyMatrix struct
impl PolyMatrix {
    /// Create a new polynomial matrix given a slice of matrix coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of matrix coefficients
    pub(crate) fn new_from_coeffs(matr_coeffs: &[DMatrix<f64>]) -> Self {
        let shape = matr_coeffs[0].shape();
        assert!(matr_coeffs.iter().all(|c| c.shape() == shape));
        PolyMatrix {
            matr_coeffs: matr_coeffs.into(),
        }
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
    pub fn right_mul(&self, rhs: &DMatrix<f64>) -> Self {
        let res: Vec<_> = self.matr_coeffs.iter().map(|x| x * rhs).collect();
        Self::new_from_coeffs(&res)
    }

    /// Implementation of matrix and polynomial matrix multiplication
    ///
    /// DMatrix * PolyMatrix
    pub fn left_mul(&self, lhs: &DMatrix<f64>) -> Self {
        let res: Vec<_> = self.matr_coeffs.iter().map(|r| lhs * r).collect();
        Self::new_from_coeffs(&res)
    }
}

impl Eval<DMatrix<Complex64>> for PolyMatrix {
    fn eval(&self, s: DMatrix<Complex64>) -> DMatrix<Complex64> {
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

        let mut res = DMatrix::from_element(rows, cols, Complex64::zero());

        for mc in self.matr_coeffs.iter().rev() {
            let mcplx = mc.map(|x| Complex64::new(x, 0.0));
            res = res.component_mul(&s) + mcplx;
        }
        res
    }
}

/// Implementation of polynomial matices addition
impl Add<PolyMatrix> for PolyMatrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // Check which polynomial matrix has the highest degree
        let new_coeffs = if self.degree() < rhs.degree() {
            let mut res = rhs.matr_coeffs.to_vec();
            for (i, c) in self.matr_coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else if rhs.degree() < self.degree() {
            let mut res = self.matr_coeffs.to_owned();
            for (i, c) in rhs.matr_coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else {
            zip_with(&self.matr_coeffs, &rhs.matr_coeffs, |l, r| l + r)
        };
        PolyMatrix::new_from_coeffs(&new_coeffs)
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

/// Implemantation of polynomial matrix printing
impl fmt::Display for PolyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.degree() == 0 {
            return write!(f, "{}", self.matr_coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.matr_coeffs.iter().enumerate() {
            if c.iter().all(|&x| x == 0.0) {
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
