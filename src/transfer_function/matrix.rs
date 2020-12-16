//! # Matrix of transfer functions
//!
//! `TfMatrix` allow the definition of a matrix of transfer functions. The
//! numerators are stored in a matrix, while the denominator is stored once,
//! since it is equal for every transfer function.
//! * evaluation of a vector of inputs
//! * conversion from a generic state space representation

use ndarray::{Array2, Axis, Zip};
use num_complex::Complex;
use num_traits::{Float, MulAdd, One, Signed, Zero};

use std::ops::{Index, IndexMut};
use std::{
    fmt,
    fmt::{Debug, Display},
};

use crate::{
    complex,
    enums::Time,
    linear_system::{self, SsGen},
    polynomial::Poly,
    polynomial_matrix::{MatrixOfPoly, PolyMatrix},
};

/// Matrix of transfer functions
#[derive(Debug)]
pub struct TfMatrix<T> {
    /// Polynomial matrix of the numerators
    num: MatrixOfPoly<T>,
    /// Common polynomial denominator
    den: Poly<T>,
}

/// Implementation of transfer function matrix
impl<T> TfMatrix<T> {
    /// Create a new transfer function matrix
    ///
    /// # Arguments
    ///
    /// * `num` - Polynomial matrix
    /// * `den` - Characteristic polynomial of the system
    fn new(num: MatrixOfPoly<T>, den: Poly<T>) -> Self {
        Self { num, den }
    }
}

impl<T: Clone> TfMatrix<T> {
    /// Retrive the characteristic polynomial of the system.
    #[must_use]
    pub fn den(&self) -> Poly<T> {
        self.den.clone()
    }
}

impl<T: Float + MulAdd<Output = T>> TfMatrix<T> {
    /// Evaluate the matrix transfers function.
    ///
    /// # Arguments
    ///
    /// * `s` - Array at which the Matrix of transfer functions is evaluated.
    #[must_use]
    pub fn eval(&self, s: &[Complex<T>]) -> Vec<Complex<T>> {
        //
        // ┌  ┐ ┌┌         ┐ ┌     ┐┐┌  ┐
        // │y1│=││1/pc 1/pc│*│n1 n2│││s1│
        // │y2│ ││1/pc 1/pc│ │n3 n4│││s2│
        // └  ┘ └└         ┘ └     ┘┘└  ┘
        // `*` is the element by element multiplication
        // ┌     ┐ ┌┌         ┐ ┌     ┐┐ ┌┌     ┐ ┌     ┐┐
        // │y1+y2│=││1/pc 1/pc│.│s1 s2││*││n1 n2│.│s1 s2││
        // │y3+y4│ ││1/pc 1/pc│ │s1 s2││ ││n3 n4│ │s1 s2││
        // └     ┘ └└         ┘ └     ┘┘ └└     ┘ └     ┘┘
        // `.` means 'evaluated at'

        // Create a matrix to contain the result of the evaluation.
        let mut res = Array2::from_elem(
            self.num.matrix().dim(),
            Complex::<T>::new(T::zero(), T::zero()),
        );

        // Zip the result and the numerator matrix row by row.
        Zip::from(res.genrows_mut())
            .and(self.num.matrix().genrows())
            .apply(|mut res_row, matrix_row| {
                // Zip the row of the result matrix.
                Zip::from(&mut res_row)
                    .and(s) // The vector of the input.
                    .and(matrix_row) // The row of the numerator matrix.
                    .apply(|r, s, n| *r = complex::compdiv(n.eval(s), self.den.eval(s)));
            });

        res.sum_axis(Axis(1)).to_vec()
    }
}

impl<T: Time> From<SsGen<f64, T>> for TfMatrix<f64> {
    /// Convert a state-space representation into a matrix of transfer functions
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn from(ss: SsGen<f64, T>) -> Self {
        let (pc, a_inv) = linear_system::leverrier_f64(&ss.a);
        let g = a_inv.left_mul(&ss.c).right_mul(&ss.b);
        let rest = PolyMatrix::multiply(&pc, &ss.d);
        let tf = g + rest;
        Self::new(MatrixOfPoly::from(tf), pc)
    }
}

impl<T: Time> From<SsGen<f32, T>> for TfMatrix<f32> {
    /// Convert a state-space representation into a matrix of transfer functions
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn from(ss: SsGen<f32, T>) -> Self {
        let (pc, a_inv) = linear_system::leverrier_f32(&ss.a);
        let g = a_inv.left_mul(&ss.c).right_mul(&ss.b);
        let rest = PolyMatrix::multiply(&pc, &ss.d);
        let tf = g + rest;
        Self::new(MatrixOfPoly::from(tf), pc)
    }
}

/// Implement read only indexing of the numerator of a transfer function matrix.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T> Index<[usize; 2]> for TfMatrix<T> {
    type Output = Poly<T>;

    fn index(&self, i: [usize; 2]) -> &Poly<T> {
        &self.num[i]
    }
}

/// Implement mutable indexing of the numerator of a transfer function matrix
/// returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T> IndexMut<[usize; 2]> for TfMatrix<T> {
    fn index_mut(&mut self, i: [usize; 2]) -> &mut Poly<T> {
        &mut self.num[i]
    }
}

/// Implementation of transfer function matrix printing
impl<T: Display + One + PartialEq + Signed + Zero> fmt::Display for TfMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s_num = self.num.to_string();
        let s_den = self.den.to_string();

        let length = s_den.len();
        let dash = "\u{2500}".repeat(length);

        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, Ss, Ssd};

    #[test]
    fn tf_matrix_new() {
        let sys = Ssd::new_from_slice(
            2,
            2,
            2,
            &[-2., 0., 0., -1.],
            &[0., 1., 1., 2.],
            &[1., 2., 3., 4.],
            &[1., 0., 0., 1.],
        );
        let tfm = TfMatrix::<f32>::from(sys);
        assert_eq!(tfm[[0, 0]], poly!(6., 5., 1.));
        assert_eq!(tfm[[0, 1]], poly!(9., 5.));
        assert_eq!(tfm[[1, 0]], poly!(8., 4.));
        assert_eq!(tfm[[1, 1]], poly!(21., 14., 1.));
        assert_eq!(tfm.den(), poly!(2., 3., 1.));
    }

    #[test]
    fn tf_matrix_eval() {
        let sys = Ss::new_from_slice(
            2,
            2,
            2,
            &[-2., 0., 0., -1.],
            &[0., 1., 1., 2.],
            &[1., 2., 3., 4.],
            &[1., 0., 0., 1.],
        );
        let tfm = TfMatrix::from(sys);
        let i = Complex::<f64>::i();
        let res = tfm.eval(&[i, i]);
        assert_relative_eq!(res[0].re, 4.4, max_relative = 1e-15);
        assert_relative_eq!(res[0].im, -3.2, max_relative = 1e-15);
        assert_relative_eq!(res[1].re, 8.2, max_relative = 1e-15);
        assert_relative_eq!(res[1].im, -6.6, max_relative = 1e-15);
    }

    #[test]
    fn tf_matrix_index_mut() {
        let sys = Ss::new_from_slice(
            2,
            2,
            2,
            &[-2., 0., 0., -1.],
            &[0., 1., 1., 2.],
            &[1., 2., 3., 4.],
            &[1., 0., 0., 1.],
        );
        let mut tfm = TfMatrix::from(sys);
        tfm[[0, 0]] = poly!(1., 2., 3.);
        assert_eq!(poly!(1., 2., 3.), tfm[[0, 0]]);
    }

    #[test]
    fn tf_matrix_print() {
        let sys = Ss::new_from_slice(
            2,
            2,
            2,
            &[-2., 0., 0., -1.],
            &[0., 1., 1., 2.],
            &[1., 2., 3., 4.],
            &[1., 0., 0., 1.],
        );
        let tfm = TfMatrix::from(sys);
        assert_eq!(
            "[[6 +5s +1s^2, 9 +5s],\n [8 +4s, 21 +14s +1s^2]]\n\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n2 +3s +1s^2",
            format!("{}", tfm)
        );
    }
}
