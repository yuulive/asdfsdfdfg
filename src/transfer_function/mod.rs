//! # Transfer function and matrices of transfer functions
//!
//! `Tf` contains the numerator and the denominator separately. Zeroes an poles
//! can be calculated.
//!
//! `TfMatrix` allow the definition of a matrix of transfer functions. The
//! numerators are stored in a matrix, while the denominator is stored once,
//! since it is equal for every transfer function.

pub mod discrete_tf;

use crate::{
    linear_system::{self, Ss},
    plots::{
        bode::{BodeIterator, BodePlot},
        polar::{PolarIterator, PolarPlot},
    },
    polynomial::{MatrixOfPoly, Poly},
    units::{Decibel, RadiantsPerSecond},
    Eval,
};

use nalgebra::{ComplexField, RealField, Scalar};
use ndarray::{Array2, Axis, Zip};
use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd, One, Signed, Zero};

use std::convert::TryFrom;
use std::ops::{Index, IndexMut, Mul};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

/// Transfer function representation of a linear system
#[derive(Debug, PartialEq)]
pub struct Tf<T> {
    /// Transfer function numerator
    num: Poly<T>,
    /// Transfer function denominator
    den: Poly<T>,
}

/// Implementation of transfer function methods
impl<T: Float> Tf<T> {
    /// Create a new transfer function given its numerator and denominator
    ///
    /// # Arguments
    ///
    /// * `num` - Transfer function numerator
    /// * `den` - Transfer function denominator
    pub fn new(num: Poly<T>, den: Poly<T>) -> Self {
        assert!(!num.is_zero());
        assert!(!den.is_zero());
        Self { num, den }
    }

    /// Extract transfer function numerator
    pub fn num(&self) -> &Poly<T> {
        &self.num
    }

    /// Extract transfer function denominator
    pub fn den(&self) -> &Poly<T> {
        &self.den
    }
}

impl<T: ComplexField + Debug + Float + RealField + Scalar> Tf<T> {
    /// Calculate the poles of the transfer function
    pub fn poles(&self) -> Option<Vec<T>> {
        self.den.roots()
    }

    /// Calculate the poles of the transfer function
    pub fn complex_poles(&self) -> Vec<Complex<T>> {
        self.den.complex_roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn zeros(&self) -> Option<Vec<T>> {
        self.num.roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn complex_zeros(&self) -> Vec<Complex<T>> {
        self.num.complex_roots()
    }
}

impl TryFrom<Ss<f64>> for Tf<f64> {
    type Error = &'static str;

    /// Convert a state-space representation into transfer functions.
    /// Conversion is available for Single Input Single Output system.
    /// If fails if the system is not SISO
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn try_from(ss: Ss<f64>) -> Result<Self, Self::Error> {
        let (pc, a_inv) = linear_system::leverrier(ss.a());
        let g = a_inv.left_mul(ss.c()).right_mul(ss.b());
        let rest = pc.multiply(ss.d());
        let tf = g + rest;
        if let Some(num) = MatrixOfPoly::from(tf).siso() {
            Ok(Self::new(num.clone(), pc))
        } else {
            Err("Linear system is not Single Input Single Output")
        }
    }
}

/// Implementation of transfer function multiplication
impl<T: Copy + Float + Mul<Output = T> + PartialEq + Zero> Mul for &Tf<T> {
    type Output = Tf<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let num = &self.num * &rhs.num;
        let den = &self.den * &rhs.den;
        Tf::new(num, den)
    }
}

/// Implementation of transfer function multiplication
impl<T: Copy + Float + Mul<Output = T> + PartialEq + Zero> Mul for Tf<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Mul::mul(&self, &rhs)
    }
}

/// Implementation of the evaluation of a transfer function
impl<T: Float + MulAdd<Output = T>> Eval<Complex<T>> for Tf<T> {
    fn eval(&self, s: &Complex<T>) -> Complex<T> {
        self.num.eval(s) / self.den.eval(s)
    }
}

/// Implementation of the Bode plot for a transfer function
impl<T: Decibel<T> + Float + FloatConst + MulAdd<Output = T>> BodePlot<T> for Tf<T> {
    fn bode(
        self,
        min_freq: RadiantsPerSecond<T>,
        max_freq: RadiantsPerSecond<T>,
        step: T,
    ) -> BodeIterator<T> {
        BodeIterator::new(self, min_freq, max_freq, step)
    }
}

/// Implementation of the polar plot for a transfer function
impl<T: Float + FloatConst + MulAdd<Output = T>> PolarPlot<T> for Tf<T> {
    fn polar(
        self,
        min_freq: RadiantsPerSecond<T>,
        max_freq: RadiantsPerSecond<T>,
        step: T,
    ) -> PolarIterator<T> {
        PolarIterator::new(self, min_freq, max_freq, step)
    }
}

/// Implementation of transfer function printing
impl<T: Display + One + PartialEq + Signed + Zero> Display for Tf<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s_num = self.num.to_string();
        let s_den = self.den.to_string();

        let length = s_num.len().max(s_den.len());
        let dash = "\u{2500}".repeat(length);

        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}

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
    pub(crate) fn new(num: MatrixOfPoly<T>, den: Poly<T>) -> Self {
        Self { num, den }
    }
}

impl<T: Clone> TfMatrix<T> {
    /// Retrive the characteristic polynomial of the system.
    pub fn den(&self) -> Poly<T> {
        self.den.clone()
    }
}

impl<T: Float + MulAdd<Output = T>> Eval<Vec<Complex<T>>> for TfMatrix<T> {
    fn eval(&self, s: &Vec<Complex<T>>) -> Vec<Complex<T>> {
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
            self.num.matrix.dim(),
            Complex::<T>::new(T::zero(), T::zero()),
        );

        // Zip the result and the numerator matrix row by row.
        Zip::from(res.genrows_mut())
            .and(self.num.matrix.genrows())
            .apply(|mut res_row, matrix_row| {
                // Zip the row of the result matrix.
                Zip::from(&mut res_row)
                    .and(s) // The vector of the input.
                    .and(matrix_row) // The row of the numerator matrix.
                    .apply(|r, s, n| *r = n.eval(s).fdiv(self.den.eval(s)));
            });

        res.sum_axis(Axis(1)).to_vec()
    }
}

impl From<Ss<f64>> for TfMatrix<f64> {
    /// Convert a state-space representation into a matrix of transfer functions
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn from(ss: Ss<f64>) -> Self {
        let (pc, a_inv) = linear_system::leverrier(ss.a());
        let g = a_inv.left_mul(ss.c()).right_mul(ss.b());
        let rest = pc.multiply(ss.d());
        let tf = g + rest;
        Self::new(MatrixOfPoly::from(tf), pc)
    }
}

/// Implement read only indexing of transfer function matrix.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T> Index<[usize; 2]> for TfMatrix<T> {
    type Output = Poly<T>;

    fn index(&self, i: [usize; 2]) -> &Poly<T> {
        &self.num.matrix[i]
    }
}

/// Implement mutable indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
impl<T> IndexMut<[usize; 2]> for TfMatrix<T> {
    fn index_mut(&mut self, i: [usize; 2]) -> &mut Poly<T> {
        &mut self.num.matrix[i]
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
    use crate::poly;

    #[test]
    fn transfer_function_creation() {
        let num = poly!(1., 2., 3.);
        let den = poly!(-4.2, -3.12, 0.0012);
        let tf = Tf::new(num.clone(), den.clone());
        assert_eq!(&num, tf.num());
        assert_eq!(&den, tf.den());
    }

    #[test]
    fn poles() {
        let tf = Tf::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(Some(vec![2., 3.]), tf.poles());
    }

    #[test]
    fn complex_poles() {
        use num_complex::Complex32;
        let tf = Tf::new(poly!(1.), poly!(10., -6., 1.));
        assert_eq!(
            vec![Complex32::new(3., -1.), Complex32::new(3., 1.)],
            tf.complex_poles()
        );
    }

    #[test]
    fn zeros() {
        let tf = Tf::new(poly!(1.), poly!(6., -5., 1.));
        assert_eq!(Some(vec![]), tf.zeros());
    }

    #[test]
    fn complex_zeros() {
        use num_complex::Complex32;
        let tf = Tf::new(poly!(3.25, 3., 1.), poly!(10., -3., 1.));
        assert_eq!(
            vec![Complex32::new(-1.5, -1.), Complex32::new(-1.5, 1.)],
            tf.complex_zeros()
        );
    }

    #[test]
    fn tf_mul() {
        let tf1 = Tf::new(poly!(1., 2., 3.), poly!(1., 5.));
        let tf2 = Tf::new(poly!(3.), poly!(1., 6., 5.));
        let actual = &tf1 * &tf2;
        let expected = Tf::new(poly!(3., 6., 9.), poly!(1., 11., 35., 25.));
        assert_eq!(expected, actual);
    }

    #[test]
    fn bode() {
        let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        let b = tf.bode(RadiantsPerSecond(0.1), RadiantsPerSecond(100.0), 0.1);
        for g in b.into_db_deg() {
            assert!(g.magnitude() < 0.);
            assert!(g.phase() < 0.);
        }
    }

    #[test]
    fn polar() {
        let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));
        let p = tf.polar(RadiantsPerSecond(0.1), RadiantsPerSecond(10.0), 0.1);
        for g in p {
            assert!(g.magnitude() < 1.);
            assert!(g.phase() < 0.);
        }
    }

    #[test]
    fn print() {
        let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        assert_eq!("1\n──────\n1 +1*s", format!("{}", tf));
    }

    #[test]
    fn tf_matrix_new() {
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
        assert_eq!(tfm[[0, 0]], poly!(6., 5., 1.));
        assert_eq!(tfm[[0, 1]], poly!(9., 5.));
        assert_eq!(tfm[[1, 0]], poly!(8., 4.));
        assert_eq!(tfm[[1, 1]], poly!(21., 14., 1.));
        assert_eq!(tfm.den, poly!(2., 3., 1.));
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
        let res = tfm.eval(&vec![i, i]);
        assert_relative_eq!(res[0].re, 4.4, max_relative = 1e-15);
        assert_relative_eq!(res[0].im, -3.2, max_relative = 1e-15);
        assert_relative_eq!(res[1].re, 8.2, max_relative = 1e-15);
        assert_relative_eq!(res[1].im, -6.6, max_relative = 1e-15);
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
            "[[6 +5*s +1*s^2, 9 +5*s],\n [8 +4*s, 21 +14*s +1*s^2]]\n─────────────\n2 +3*s +1*s^2",
            format!("{}", tfm)
        );
    }
}
