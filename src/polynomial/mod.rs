use crate::Eval;

use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use nalgebra::{DMatrix, Schur};
use ndarray::{Array, Array2, Zip};
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
    /// of the monic polynomial.
    fn companion(&self) -> DMatrix<f64> {
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
}

/// Evaluate the polynomial at the given float number
impl Eval<Complex64> for Poly {
    fn eval(&self, x: &Complex64) -> Complex64 {
        self.coeffs
            .iter()
            .rev()
            .fold(Complex::zero(), |acc, &c| acc * x + c)
    }
}

/// Evaluate the polynomial at the given complex number
impl Eval<f64> for Poly {
    fn eval(&self, x: &f64) -> f64 {
        self.coeffs
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc.mul_add(*x, c))
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
            crate::zip_with(&self.coeffs, &rhs.coeffs, |l, r| l + r)
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

/// Implemantation of polynomial and matrix multiplication
impl Mul<&DMatrix<f64>> for &Poly {
    type Output = PolyMatrix;

    fn mul(self, rhs: &DMatrix<f64>) -> PolyMatrix {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let res: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
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
    fn poly_f64_eval_test() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(86., p.eval(&5.));

        assert_eq!(0.0, Poly::new_from_coeffs(&[]).eval(&6.4));
    }

    #[test]
    fn poly_cmplx_eval_test() {
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
    pub(crate) matr_coeffs: Vec<DMatrix<f64>>,
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
        let mut pm = PolyMatrix {
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
        let res: Vec<_> = self.matr_coeffs.iter().map(|x| x * rhs).collect();
        Self::new_from_coeffs(&res)
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

        let mut res = DMatrix::from_element(rows, cols, Complex64::zero());

        for mc in self.matr_coeffs.iter().rev() {
            let mcplx = mc.map(|x| Complex64::new(x, 0.0));
            res = res.component_mul(s) + mcplx;
        }
        res
    }
}

/// Implementation of polynomial matrices addition
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
            crate::zip_with(&self.matr_coeffs, &rhs.matr_coeffs, |l, r| l + r)
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

/// Polynomial matrix object
///
/// Contains the matrix of polynomials
///
/// P(x) = [[P1, P2], [P3, P4]]
#[derive(Debug)]
pub struct MP {
    pub(crate) matrix: Array2<Poly>,
}

/// Implementation methods for MP struct
impl MP {
    /// Create a new polynomial matrix given a vector of polynomials.
    ///
    /// # Arguments
    ///
    /// * `rows` - number of rows of the matrix
    /// * `cols` - number of colums of the matrix
    /// * `data` - vector of polynomials in row major order
    ///
    /// # Panics
    ///
    /// Panics if the matrix cannot be build from given arguments.
    fn new(rows: usize, cols: usize, data: Vec<Poly>) -> Self {
        Self {
            matrix: Array::from_shape_vec((rows, cols), data)
                .expect("Input data do not allow to create the matrix"),
        }
    }

    /// Extract the transfer function from the matrix if is the only one.
    /// Use to get Single Input Single Output transfer function.
    pub fn siso(&self) -> Option<&Poly> {
        if self.matrix.shape() == [1, 1] {
            self.matrix.first()
        } else {
            None
        }
    }
}

/// Implement conversion between different representations.
impl From<PolyMatrix> for MP {
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

        // Crate a vecor containing the vector of coefficients a single
        // polynomial in row major mode with respect to the initial
        // vector of matrices.
        let mut tmp: Vec<Vec<f64>> = vec![vec![]; rows * cols];
        for order in vectorized_coeffs {
            for (i, value) in order.into_iter().enumerate() {
                tmp[i].push(value);
            }
        }

        let polys: Vec<Poly> = tmp.iter().map(|p| Poly::new_from_coeffs(&p)).collect();
        MP::new(rows, cols, polys)
    }
}

impl Eval<Array2<Complex64>> for MP {
    fn eval(&self, s: &Array2<Complex64>) -> Array2<Complex64> {
        // transform matr_coeffs in complex numbers matrices
        //
        // ┌     ┐ ┌     ┐ ┌     ┐
        // │c1 c2│=│P1 P2│*│s1 s2│
        // │c3 c4│ │P3 P4│ │s1 s2│
        // └     ┘ └     ┘ └     ┘
        // `*` is the element by element evaluation
        let mut res = Array2::from_elem(self.matrix.dim(), Complex::new(0.0, 0.0));
        Zip::from(&mut res)
            .and(&self.matrix)
            .and(s)
            .apply(|ci, pi, &si| *ci = pi.eval(&si));
        res
    }
}

/// Implementation of matrix of polynomials printing
impl fmt::Display for MP {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn mp_creation_test() {
        let c = [4.3, 5.32];
        let p = Poly::new_from_coeffs(&c);
        let v = vec![p.clone(), p.clone(), p.clone(), p.clone()];
        let mp = MP::new(2, 2, v);
        let expected = "[[4.3 +5.32*s, 4.3 +5.32*s],\n [4.3 +5.32*s, 4.3 +5.32*s]]";
        assert_eq!(expected, format!("{}", &mp));
    }
}
