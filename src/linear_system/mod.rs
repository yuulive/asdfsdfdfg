//! # Linear system
//!
//! This module contains the state-space representation of the linear system.
//!
//! It is possible to calculate the equilibrium point of the system.
//!
//! The time evolution of the system is defined through iterator, created by
//! different solvers.

pub mod discrete;
pub mod solver;

use crate::{
    linear_system::solver::{Order, RadauIterator, RkIterator, Rkf45Iterator},
    polynomial::{Poly, PolyMatrix},
    transfer_function::Tf,
    units::Seconds,
};

use nalgebra::{ComplexField, DMatrix, DVector, RealField, Scalar, Schur};
use num_complex::Complex;
use num_traits::Float;

use std::convert::From;
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

/// State-space representation of a linear system
///
/// ```text
/// xdot(t) = A * x(t) + B * u(t)
/// y(t)    = C * x(t) + D * u(t)
/// ```
#[derive(Debug)]
pub struct Ss<T: Scalar> {
    /// A matrix
    a: DMatrix<T>,
    /// B matrix
    b: DMatrix<T>,
    /// C matrix
    c: DMatrix<T>,
    /// D matrix
    d: DMatrix<T>,
    /// Dimensions
    dim: Dim,
}

/// Dim of the linar system.
#[derive(Debug, Clone, Copy)]
pub struct Dim {
    /// Number of states
    states: usize,
    /// Number of inputs
    inputs: usize,
    /// Number of outputs
    outputs: usize,
}

/// Implementation of the methods for the Dim struct.
impl Dim {
    /// Get the number of states.
    pub fn states(&self) -> usize {
        self.states
    }

    /// Get the number of inputs.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Get the number of outputs.
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

/// Implementation of the methods for the state-space
impl<T: Scalar> Ss<T> {
    /// Create a new state-space representation
    ///
    /// # Arguments
    ///
    /// * `states` - number of states (n)
    /// * `inputs` - number of inputs (m)
    /// * `outputs` - number of outputs (p)
    /// * `a` - A matrix (nxn)
    /// * `b` - B matrix (nxm)
    /// * `c` - C matrix (pxn)
    /// * `d` - D matrix (pxm)
    ///
    /// # Panics
    ///
    /// Panics if matrix dimensions do not match
    pub fn new_from_slice(
        states: usize,
        inputs: usize,
        outputs: usize,
        a: &[T],
        b: &[T],
        c: &[T],
        d: &[T],
    ) -> Self {
        Self {
            a: DMatrix::from_row_slice(states, states, a),
            b: DMatrix::from_row_slice(states, inputs, b),
            c: DMatrix::from_row_slice(outputs, states, c),
            d: DMatrix::from_row_slice(outputs, inputs, d),
            dim: Dim {
                states,
                inputs,
                outputs,
            },
        }
    }

    /// Get the A matrix
    pub(crate) fn a(&self) -> &DMatrix<T> {
        &self.a
    }

    /// Get the C matrix
    pub(crate) fn b(&self) -> &DMatrix<T> {
        &self.b
    }

    /// Get the C matrix
    pub(crate) fn c(&self) -> &DMatrix<T> {
        &self.c
    }

    /// Get the D matrix
    pub(crate) fn d(&self) -> &DMatrix<T> {
        &self.d
    }

    /// Get the dimensions of the system (states, inputs, outputs).
    pub fn dim(&self) -> Dim {
        self.dim
    }
}

/// Implementation of the methods for the state-space
impl<T: ComplexField + RealField> Ss<T> {
    /// Calculate the poles of the system
    pub fn poles(&self) -> Vec<Complex<T>> {
        Schur::new(self.a.clone())
            .complex_eigenvalues()
            .as_slice()
            .to_vec()
    }
}

/// Implementation of the methods for the state-space
impl<T: ComplexField + Scalar> Ss<T> {
    /// Calculate the equilibrium point for the given input condition
    ///
    /// # Arguments
    ///
    /// * `u` - Input vector
    pub fn equilibrium(&self, u: &[T]) -> Option<Equilibrium<T>> {
        assert_eq!(u.len(), self.b.ncols(), "Wrong number of inputs.");
        let u = DVector::from_row_slice(u);
        let inv_a = &self.a.clone().try_inverse()?;
        let x = -inv_a * &self.b * &u;
        let y = (-&self.c * inv_a * &self.b + &self.d) * u;
        Some(Equilibrium::new(x, y))
    }
}

/// Implementation of the methods for the state-space
impl Ss<f64> {
    /// Time evolution for the given input, using Runge-Kutta second order method
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column mayor)
    /// * `x0` - initial state (column mayor)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub fn rk2<F>(&self, u: F, x0: &[f64], h: Seconds<f64>, n: usize) -> RkIterator<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        RkIterator::new(self, u, x0, h, n, Order::Rk2)
    }

    /// Time evolution for the given input, using Runge-Kutta fourth order method
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column mayor)
    /// * `x0` - initial state (column mayor)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub fn rk4<F>(&self, u: F, x0: &[f64], h: Seconds<f64>, n: usize) -> RkIterator<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        RkIterator::new(self, u, x0, h, n, Order::Rk4)
    }

    /// Runge-Kutta-Fehlberg 45 with adaptive step for time evolution.
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `limit` - time evaluation limit
    /// * `tol` - error tolerance
    pub fn rkf45<F>(
        &self,
        u: F,
        x0: &[f64],
        h: Seconds<f64>,
        limit: Seconds<f64>,
        tol: f64,
    ) -> Rkf45Iterator<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        Rkf45Iterator::new(self, u, x0, h, limit, tol)
    }

    /// Radau of order 3 with 2 steps method for time evolution.
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    /// * `tol` - error tolerance
    pub fn radau<F>(
        &self,
        u: F,
        x0: &[f64],
        h: Seconds<f64>,
        n: usize,
        tol: f64,
    ) -> RadauIterator<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        RadauIterator::new(self, u, x0, h, n, tol)
    }
}

/// Faddeev–LeVerrier algorithm
///
/// (https://en.wikipedia.org/wiki/Faddeev%E2%80%93LeVerrier_algorithm)
///
/// B(s) =       B1*s^(n-1) + B2*s^(n-2) + B3*s^(n-3) + ...
/// a(s) = s^n + a1*s^(n-1) + a2*s^(n-2) + a3*s^(n-3) + ...
///
/// with B1 = I = eye(n,n)
/// a1 = -trace(A); ak = -1/k * trace(A*Bk)
/// Bk = a_(k-1)*I + A*B_(k-1)
#[allow(non_snake_case, clippy::cast_precision_loss)]
pub(crate) fn leverrier(A: &DMatrix<f64>) -> (Poly<f64>, PolyMatrix<f64>) {
    let size = A.nrows(); // A is a square matrix.
    let mut a = vec![1.0];
    let a1 = -A.trace();
    a.insert(0, a1);

    let B1 = DMatrix::identity(size, size); // eye(n,n)
    let mut B = vec![B1.clone()];
    if size == 1 {
        return (Poly::new_from_coeffs(&a), PolyMatrix::new_from_coeffs(&B));
    }

    let mut Bk = B1.clone();
    let mut ak = a1;
    for k in 2..=size {
        Bk = ak * &B1 + A * &Bk;
        B.insert(0, Bk.clone());

        let ABk = A * &Bk;
        // Casting usize to f64 causes a loss of precision on targets with
        // 64-bit wide pointers (usize is 64 bits wide, but f64's mantissa is
        // only 52 bits wide)
        ak = -(k as f64).recip() * ABk.trace();
        a.insert(0, ak);
    }
    (Poly::new_from_coeffs(&a), PolyMatrix::new_from_coeffs(&B))
}

impl<T: Float + Scalar + ComplexField + RealField> From<Tf<T>> for Ss<T> {
    /// Convert a transfer function representation into state space representation.
    /// Conversion is done using the observability canonical form.
    ///
    /// ```text
    ///        b_n*s^n + b_(n-1)*s^(n-1) + ... + b_1*s + b_0
    /// G(s) = ---------------------------------------------
    ///          s^n + a_(n-1)*s^(n-1) + ... + a_1*s + a_0
    ///     ┌                   ┐        ┌         ┐
    ///     │ 0 0 0 . 0 -a_0    │        │ b'_0    │
    ///     │ 1 0 0 . 0 -a_1    │        │ b'_1    │
    /// A = │ 0 1 0 . 0 -a_2    │,   B = │ b'_2    │
    ///     │ . . . . . .       │        │ .       │
    ///     │ 0 0 0 . 1 -a_(n-1)│        │ b'_(n-1)│
    ///     └                   ┘        └         ┘
    ///     ┌           ┐                ┌    ┐
    /// C = │0 0 0 . 0 1│,           D = │b'_n│
    ///     └           ┘                └    ┘
    ///
    /// b'_n = b_n,   b'_i = b_i - a_i*b'_n,   i = 0, ..., n-1
    /// ```
    /// A is the companion matrix of the transfer function denominator.
    /// The denominator is in monic form, this means that the numerator shall be
    /// divided by the leading coefficient of the original denominator.
    ///
    /// # Arguments
    ///
    /// `tf` - transfer function
    fn from(tf: Tf<T>) -> Self {
        // Get the denominator in the monic form and the leading coefficient.
        let (den_monic, den_n) = tf.den().monic();
        // Extend the numerator coefficients with zeros to the length of the
        // denominator polynomial.
        let order = den_monic.degree();
        // Divide the denominator polynomial by the highest coefficient of the
        // numerator polinomial to mantain the original gain.
        let mut num = tf.num().clone() / den_n;
        num.extend(order);

        // Calculate the observability canonical form.
        let a = den_monic.companion();

        // Get the number of states n.
        let states = a.nrows();
        // Get the highest coefficient of the numerator.
        let b_n = num[order];

        // Create a nx1 vector with b'i = bi - ai * b'n
        let b = DMatrix::from_fn(states, 1, |i, _j| num[i] - den_monic[i] * b_n);

        // Crate a 1xn vector with all zeros but the last that is 1.
        let mut c = DMatrix::zeros(1, states);
        c[states - 1] = T::one();

        // Crate a 1x1 matrix with the highest coefficient of the numerator.
        let d = DMatrix::from_element(1, 1, b_n);

        // A single transfer function has only one input and one output.
        Self {
            a,
            b,
            c,
            d,
            dim: Dim {
                states,
                inputs: 1,
                outputs: 1,
            },
        }
    }
}

/// Implementation of state-space representation
impl<T: Scalar + Display> Display for Ss<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "A: {}\nB: {}\nC: {}\nD: {}",
            self.a, self.b, self.c, self.d
        )
    }
}

/// Struct describing an equilibrium point
#[derive(Debug)]
pub struct Equilibrium<T: Scalar> {
    /// State equilibrium
    x: DVector<T>,
    /// Output equilibrium
    y: DVector<T>,
}

/// Implement methods for equilibrium
impl<T: Scalar> Equilibrium<T> {
    /// Create a new equilibrium given the state and the output vectors
    ///
    /// # Arguments
    ///
    /// * `x` - State equilibrium
    /// * `y` - Output equilibrium
    pub(crate) fn new(x: DVector<T>, y: DVector<T>) -> Self {
        Self { x, y }
    }

    /// Retrieve state coordinates for equilibrium
    pub fn x(&self) -> &[T] {
        self.x.as_slice()
    }

    /// Retrieve output coordinates for equilibrium
    pub fn y(&self) -> &[T] {
        self.y.as_slice()
    }
}

/// Implementation of printing of equilibrium point
impl<T: Display + Scalar> Display for Equilibrium<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "x: {}\ny: {}", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn leverrier_algorythm() {
        use crate::polynomial::MatrixOfPoly;

        // Example of LeVerrier algorithm (Wikipedia)");
        let t = DMatrix::from_row_slice(3, 3, &[3., 1., 5., 3., 3., 1., 4., 6., 4.]);
        let expected_pc = Poly::new_from_coeffs(&[-40., 4., -10., 1.]);
        let expected_degree0 =
            DMatrix::from_row_slice(3, 3, &[6., 26., -14., -8., -8., 12., 6., -14., 6.]);
        let expected_degree1 =
            DMatrix::from_row_slice(3, 3, &[-7., 1., 5., 3., -7., 1., 4., 6., -6.]);
        let expected_degree2 = DMatrix::from_row_slice(3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);

        let (p, poly_matrix) = leverrier(&t);

        println!("T: {}\np: {}\n", &t, &p);
        println!("B: {}", &poly_matrix);

        assert_eq!(expected_pc, p);
        assert_eq!(expected_degree0, poly_matrix[0]);
        assert_eq!(expected_degree1, poly_matrix[1]);
        assert_eq!(expected_degree2, poly_matrix[2]);

        let mp = MatrixOfPoly::from(poly_matrix);
        println!("mp {}", &mp);
        let expected_result = "[[6 -7*s +1*s^2, 26 +1*s, -14 +5*s],\n \
                               [-8 +3*s, -8 -7*s +1*s^2, 12 +1*s],\n \
                               [6 +4*s, -14 +6*s, 6 -6*s +1*s^2]]";
        assert_eq!(expected_result, format!("{}", &mp));
    }

    #[test]
    fn convert_to_ss_1() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[1.]),
            Poly::new_from_coeffs(&[1., 1., 1.]),
        );

        let ss = Ss::from(tf);

        assert_eq!(DMatrix::from_row_slice(2, 2, &[0., -1., 1., -1.]), *ss.a());
        assert_eq!(DMatrix::from_row_slice(2, 1, &[1., 0.]), *ss.b());
        assert_eq!(DMatrix::from_row_slice(1, 2, &[0., 1.]), *ss.c());
        assert_eq!(DMatrix::from_row_slice(1, 1, &[0.]), *ss.d());
    }

    #[test]
    fn convert_to_ss_2() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[1., 0., 1.]),
            Poly::new_from_coeffs(&[3., 4., 1.]),
        );

        let ss = Ss::from(tf);

        assert_eq!(DMatrix::from_row_slice(2, 2, &[0., -3., 1., -4.]), *ss.a());
        assert_eq!(DMatrix::from_row_slice(2, 1, &[-2., -4.]), *ss.b());
        assert_eq!(DMatrix::from_row_slice(1, 2, &[0., 1.]), *ss.c());
        assert_eq!(DMatrix::from_row_slice(1, 1, &[1.]), *ss.d());
    }
}
