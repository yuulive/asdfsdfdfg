//! # Matrices of polynomials
//!
//! `MatrixOfPoly` allows the definition of matrices of polynomials.

use nalgebra::{DMatrix, Scalar};
use ndarray::{Array, Array2};
use num_complex::Complex;
use num_traits::{Float, NumAssignOps, One, Signed, Zero};

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

use crate::polynomial::Poly;

/// Polynomial matrix object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// P(x) = C0 + C1*x + C2*x^2 + ...
#[derive(Clone, Debug)]
pub(crate) struct PolyMatrix<T: Scalar> {
    matr_coeffs: Vec<DMatrix<T>>,
}

/// Implementation methods for `PolyMatrix` struct
impl<T: Scalar> PolyMatrix<T> {
    /// Degree of the polynomial matrix
    fn degree(&self) -> usize {
        assert!(
            !self.matr_coeffs.is_empty(),
            "Degree is not defined on empty polynomial matrix"
        );
        self.matr_coeffs.len() - 1
    }
}

/// Implementation methods for `PolyMatrix` struct
impl<T: Scalar + Zero> PolyMatrix<T> {
    /// Create a new polynomial matrix given a slice of matrix coefficients.
    ///
    /// # Arguments
    ///
    /// * `matr_coeffs` - slice of matrix coefficients
    pub(crate) fn new_from_coeffs(matr_coeffs: &[DMatrix<T>]) -> Self {
        let shape = matr_coeffs[0].shape();
        assert!(matr_coeffs.iter().all(|c| c.shape() == shape));
        let mut pm = Self {
            matr_coeffs: matr_coeffs.into(),
        };
        pm.trim();
        debug_assert!(!pm.matr_coeffs.is_empty());
        pm
    }

    /// Create a new polynomial matrix given an iterator of matrix coefficients.
    ///
    /// # Arguments
    ///
    /// * `matr_iter` - iterator of matrix coefficients
    pub(crate) fn new_from_iter<II>(matr_iter: II) -> Self
    where
        II: IntoIterator<Item = DMatrix<T>>,
    {
        let mut pm = Self {
            matr_coeffs: matr_iter.into_iter().collect(),
        };
        debug_assert!(!pm.matr_coeffs.is_empty());
        let shape = pm.matr_coeffs[0].shape();
        assert!(pm.matr_coeffs.iter().all(|c| c.shape() == shape));
        pm.trim();
        pm
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

impl<T: Scalar + Zero + One + Add + AddAssign + Mul + MulAssign> PolyMatrix<T> {
    /// Implementation of polynomial matrix and matrix multiplication
    ///
    /// `PolyMatrix` * `DMatrix`
    pub(crate) fn right_mul(&self, rhs: &DMatrix<T>) -> Self {
        let result: Vec<_> = self.matr_coeffs.iter().map(|x| x * rhs).collect();
        Self::new_from_coeffs(&result)
    }

    /// Implementation of matrix and polynomial matrix multiplication
    ///
    /// `DMatrix` * `PolyMatrix`
    pub(crate) fn left_mul(&self, lhs: &DMatrix<T>) -> Self {
        let res: Vec<_> = self.matr_coeffs.iter().map(|r| lhs * r).collect();
        Self::new_from_coeffs(&res)
    }
}

impl<T: NumAssignOps + Float + Scalar> PolyMatrix<T> {
    #[allow(dead_code)]
    /// Evaluate the polynomial matrix
    ///
    /// # Arguments
    /// * `s` - Matrix at which the polynomial matrix is evaluated.
    pub(crate) fn eval(&self, s: &DMatrix<Complex<T>>) -> DMatrix<Complex<T>> {
        // transform matr_coeffs in complex numbers matrices
        //
        // ┌     ┐ ┌       ┐ ┌       ┐ ┌     ┐ ┌       ┐ ┌         ┐
        // │P1 P2│=│a01 a02│+│a11 a12│*│s1 s2│+│a21 a22│*│s1^2 s2^2│
        // │P3 P4│ │a03 a04│ │a13 a14│ │s1 s2│ │a23 a24│ │s1^2 s2^2│
        // └     ┘ └       ┘ └       ┘ └     ┘ └       ┘ └         ┘
        // `*` is the element by element multiplication
        // If i have a 2x2 matr_coeff the result shall be a 2x2 matrix,
        // because otherwise i will sum P1 and P2 (P3 and P4)
        let rows = self.matr_coeffs[0].nrows();
        let cols = self.matr_coeffs[0].ncols();

        let mut result = DMatrix::from_element(rows, cols, Complex::<T>::zero());

        for mc in self.matr_coeffs.iter().rev() {
            let mcplx = mc.map(|x| Complex::<T>::new(x, T::zero()));
            result = result.component_mul(&s) + mcplx;
        }
        result
    }
}

/// Implementation of polynomial matrices addition
impl Add<PolyMatrix<f64>> for PolyMatrix<f64> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self {
        // Check which polynomial matrix has the highest degree
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.matr_coeffs.iter().enumerate() {
                rhs[i] += c;
            }
            rhs
        } else {
            for (i, c) in rhs.matr_coeffs.iter().enumerate() {
                self[i] += c;
            }
            self
        };
        result.trim();
        result
    }
}

impl Add<PolyMatrix<f32>> for PolyMatrix<f32> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self {
        // Check which polynomial matrix has the highest degree
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.matr_coeffs.iter().enumerate() {
                rhs[i] += c;
            }
            rhs
        } else {
            for (i, c) in rhs.matr_coeffs.iter().enumerate() {
                self[i] += c;
            }
            self
        };
        result.trim();
        result
    }
}

impl<T: Mul<Output = T> + MulAssign<T> + Scalar + Zero> PolyMatrix<T> {
    /// Implementation of polynomial and matrix multiplication
    /// It's the polynomial matrix whose coefficients are the coefficients
    /// of the polynomial times the matrix
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial
    /// * `matrix` - Matrix
    pub(crate) fn multiply(poly: &Poly<T>, matrix: &DMatrix<T>) -> PolyMatrix<T> {
        let result = poly.as_slice().iter().map(|&c| matrix * c);
        PolyMatrix::new_from_iter(result)
    }
}

/// Implementation of read only indexing of polynomial matrix
/// returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T: Scalar> Index<usize> for PolyMatrix<T> {
    type Output = DMatrix<T>;

    fn index(&self, i: usize) -> &DMatrix<T> {
        &self.matr_coeffs[i]
    }
}

/// Implementation of mutable indexing of polynomial matrix
/// returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T: Scalar> IndexMut<usize> for PolyMatrix<T> {
    fn index_mut(&mut self, i: usize) -> &mut DMatrix<T> {
        &mut self.matr_coeffs[i]
    }
}

/// Implementation of polynomial matrix printing
impl<T: Display + Scalar + Zero> Display for PolyMatrix<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.degree() == 0 {
            return write!(f, "{}", self.matr_coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.matr_coeffs.iter().enumerate() {
            if c.iter().all(|&x| x == T::zero()) {
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
/// `P(x) = [[P1, P2], [P3, P4]]`
#[derive(Debug)]
pub struct MatrixOfPoly<T> {
    pub(crate) matrix: Array2<Poly<T>>,
}

/// Implementation methods for MP struct
impl<T> MatrixOfPoly<T> {
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
    fn new(rows: usize, cols: usize, data: Vec<Poly<T>>) -> Self {
        Self {
            matrix: Array::from_shape_vec((rows, cols), data)
                .expect("Input data do not allow to create the matrix"),
        }
    }

    /// Extract the polynomial from the matrix if is the only one.
    pub(crate) fn single(&self) -> Option<&Poly<T>> {
        if self.matrix.shape() == [1, 1] {
            self.matrix.first()
        } else {
            None
        }
    }
}

/// Implement conversion between different representations.
impl<T: Scalar + Zero> From<PolyMatrix<T>> for MatrixOfPoly<T> {
    fn from(pm: PolyMatrix<T>) -> Self {
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
        let mut tmp: Vec<Vec<T>> = vec![vec![]; rows * cols];
        for order in vectorized_coeffs {
            for (i, value) in order.into_iter().enumerate() {
                tmp[i].push(value);
            }
        }

        let polys: Vec<Poly<T>> = tmp.iter().map(|p| Poly::new_from_coeffs(&p)).collect();
        Self::new(rows, cols, polys)
    }
}

/// Implementation of matrix of polynomials printing
impl<T: Display + Signed> Display for MatrixOfPoly<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_empty() {
        let v = vec![DMatrix::<f32>::zeros(1, 2), DMatrix::zeros(1, 2)];
        let pm = PolyMatrix::new_from_coeffs(&v);
        assert_eq!(DMatrix::zeros(1, 2), pm.matr_coeffs[0]);
    }

    #[test]
    fn right_mul() {
        let v = vec![
            DMatrix::from_row_slice(2, 2, &[1.0_f32, 2., 3., 4.]),
            DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]),
        ];
        let pm = PolyMatrix::new_from_coeffs(&v);
        let res = pm.right_mul(&DMatrix::from_row_slice(2, 2, &[5., 0., 0., 5.]));
        assert_eq!(
            DMatrix::from_row_slice(2, 2, &[5., 10., 15., 20.]),
            res.matr_coeffs[0]
        );
        dbg!(&res);
    }

    #[test]
    fn left_mul() {
        let v = vec![
            DMatrix::from_row_slice(2, 2, &[1.0_f32, 2., 3., 4.]),
            DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]),
        ];
        let pm = PolyMatrix::new_from_coeffs(&v);
        let res = pm.left_mul(&DMatrix::from_row_slice(2, 2, &[5., 0., 0., 5.]));
        assert_eq!(DMatrix::from_row_slice(2, 2, &[5., 10., 15., 20.]), res[0]);
    }

    #[test]
    fn eval() {
        let v = vec![
            DMatrix::from_row_slice(2, 2, &[1.0_f32, 2., 3., 4.]),
            DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]),
        ];
        let pm = PolyMatrix::new_from_coeffs(&v);
        let res = pm.eval(&DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(5., 0.),
                Complex::zero(),
                Complex::zero(),
                Complex::new(0., 5.),
            ],
        ));
        assert_eq!(
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex::new(6., 0.),
                    Complex::new(2., 0.),
                    Complex::new(3., 0.),
                    Complex::new(4., 5.)
                ]
            ),
            res
        );
    }

    #[test]
    fn add() {
        let v = vec![
            DMatrix::from_row_slice(2, 2, &[1.0_f64, 2., 3., 4.]),
            DMatrix::from_row_slice(2, 2, &[1., 0., 0., 1.]),
        ];
        let pm = PolyMatrix::new_from_coeffs(&v);
        let res = pm.clone() + pm;
        assert_eq!(DMatrix::from_row_slice(2, 2, &[2., 4., 6., 8.]), res[0]);
    }

    #[test]
    fn mp_creation() {
        let c = [4.3, 5.32];
        let p = Poly::new_from_coeffs(&c);
        let v = vec![p.clone(), p.clone(), p.clone(), p];
        let mp = MatrixOfPoly::new(2, 2, v);
        let expected = "[[4.3 +5.32s, 4.3 +5.32s],\n [4.3 +5.32s, 4.3 +5.32s]]";
        assert_eq!(expected, format!("{}", &mp));
    }

    #[test]
    fn single() {
        let v = vec![Poly::new_from_coeffs(&[4.3, 5.32])];
        let mp = MatrixOfPoly::new(1, 1, v);
        let res = mp.single();
        assert!(res.is_some());
        assert_relative_eq!(14.94, res.unwrap().eval(&2.), max_relative = 1e-10);
    }

    #[test]
    fn single_fail() {
        let c = [4.3, 5.32];
        let p = Poly::new_from_coeffs(&c);
        let v = vec![p.clone(), p.clone(), p.clone(), p];
        let mp = MatrixOfPoly::new(2, 2, v);
        let res = mp.single();
        assert!(res.is_none());
    }
}
