//! # Linear system
//!
//! This module contains the state-space representation of the linear system.
//!
//! It is possible to calculate the equilibrium point of the system.
//!
//! The time evolution of the system is defined through iterator, created by
//! different solvers.

pub mod continuous;
pub mod discrete;
pub mod solver;

use nalgebra::{ComplexField, DMatrix, DVector, RealField, Scalar, Schur};
use num_complex::Complex;
use num_traits::Float;

use std::{
    convert::TryFrom,
    fmt,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
};

use crate::{
    polynomial,
    polynomial::{matrix::PolyMatrix, Poly},
    transfer_function::TfGen,
    Time,
};

/// State-space representation of a linear system
///
/// ```text
/// xdot(t) = A * x(t) + B * u(t)
/// y(t)    = C * x(t) + D * u(t)
/// ```
#[derive(Debug)]
pub struct SsGen<T: Scalar, U: Time> {
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
    /// Tag for continuous or discrete time
    time: PhantomData<U>,
}

/// Dim of the linear system.
#[derive(Debug, Clone, Copy, PartialEq)]
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
impl<T: Scalar, U: Time> SsGen<T, U> {
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
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ss;
    /// let sys = Ss::new_from_slice(2, 1, 1, &[-2., 0., 3., -7.], &[1., 3.], &[-1., 0.5], &[0.1]);
    /// ```
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
            time: PhantomData,
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
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ssd;
    /// let sys = Ssd::new_from_slice(2, 1, 1, &[-2., 0., 3., -7.], &[1., 3.], &[-1., 0.5], &[0.1]);
    /// let dimensions = sys.dim();
    /// ```
    pub fn dim(&self) -> Dim {
        self.dim
    }
}

/// Implementation of the methods for the state-space
impl<T: ComplexField + Float + RealField, U: Time> SsGen<T, U> {
    /// Calculate the poles of the system
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ss;
    /// let sys = Ss::new_from_slice(2, 1, 1, &[-2., 0., 3., -7.], &[1., 3.], &[-1., 0.5], &[0.1]);
    /// let poles = sys.poles();
    /// assert_eq!(-2., poles[0].re);
    /// assert_eq!(-7., poles[1].re);
    /// ```
    pub fn poles(&self) -> Vec<Complex<T>> {
        if self.a.nrows() == 2 {
            let m00 = self.a[(0, 0)];
            let m01 = self.a[(0, 1)];
            let m10 = self.a[(1, 0)];
            let m11 = self.a[(1, 1)];
            let trace = m00 + m11;
            let determinant = m00 * m11 - m01 * m10;

            let (eig1, eig2) = polynomial::complex_quadratic_roots(-trace, determinant);

            vec![eig1, eig2]
        } else {
            Schur::new(self.a.clone())
                .complex_eigenvalues()
                .as_slice()
                .to_vec()
        }
    }
}

/// Controllability matrix implementation.
///
/// `Mr = [B AB A^2B ... A^(n-1)B]` -> (n, mn) matrix.
///
/// # Arguments
///
/// * `n` - Number of states
/// * `m` - Number of inputs
/// * `a` - A matrix
/// * `b` - B matrix
fn controllability_impl<T: RealField + Scalar>(
    n: usize,
    m: usize,
    a: &DMatrix<T>,
    b: &DMatrix<T>,
) -> DMatrix<T> {
    // Create the entire matrix ahead to avoid multiple allocations.
    let mut mr = DMatrix::<T>::zeros(n, n * m);
    mr.columns_range_mut(0..m).copy_from(b);
    // Create a temporary matrix for the multiplication, since the Mr matrix
    // cannot be used both as reference and mutable reference.
    let mut rhs = DMatrix::<T>::zeros(n, m);
    for i in 1..=n - 1 {
        rhs.copy_from(&mr.columns_range(((i - 1) * m)..(i * m)));
        // Multiply A by the result of the previous step.
        // The result is directly inserted into Mr.
        a.mul_to(&rhs, &mut mr.columns_range_mut((i * m)..((i + 1) * m)))
    }
    mr
}

/// Osservability matrix implementation.
///
/// `Mo = [C' A'C' A'^2B ... A'^(n-1)C']` -> (n, pn) matrix.
///
/// # Arguments
///
/// * `n` - Number of states
/// * `p` - Number of outputs
/// * `a` - A matrix
/// * `c` - C matrix
fn observability_impl<T: RealField + Scalar>(
    n: usize,
    p: usize,
    a: &DMatrix<T>,
    c: &DMatrix<T>,
) -> DMatrix<T> {
    // Create the entire matrix ahead to avoid multiple allocations.
    let mut mo = DMatrix::<T>::zeros(n, n * p);
    mo.columns_range_mut(0..p).copy_from(&c.transpose());
    // Create a temporary matrix for the multiplication, since the Mo matrix
    // cannot be used both as reference and mutable reference.
    let mut rhs = DMatrix::<T>::zeros(n, p);
    for i in 1..=n - 1 {
        rhs.copy_from(&mo.columns_range(((i - 1) * p)..(i * p)));
        // Multiply A by the result of the previous step.
        // The result is directly inserted into Mo;
        a.tr_mul_to(&rhs, &mut mo.columns_range_mut((i * p)..((i + 1) * p)))
    }
    mo
}

impl<T: RealField + Scalar, U: Time> SsGen<T, U> {
    /// Controllability matrix
    ///
    /// `Mr = [B AB A^2B ... A^(n-1)B]` -> (n, mn) matrix.
    ///
    /// The return value is: `(rows, cols, vector with data in column major mode)`
    ///
    /// # Example
    /// ```
    /// use automatica::{linear_system::SsGen, Discrete};
    /// let a = [-1., 3., 0., 2.];
    /// let b = [1., 2.];
    /// let c = [1., 1.];
    /// let d = [0.];
    /// let sys = SsGen::<_, Discrete>::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    /// let mr = sys.controllability();
    /// assert_eq!((2, 2, vec![1., 2., 5., 4.]), mr);
    /// ```
    pub fn controllability(&self) -> (usize, usize, Vec<T>) {
        let n = self.dim.states;
        let m = self.dim.inputs;
        let mr = controllability_impl(n, m, &self.a, &self.b);
        (n, n * m, mr.data.as_vec().clone())
    }

    /// Osservability matrix
    ///
    /// `Mo = [C' A'C' A'^2B ... A'^(n-1)C']` -> (n, pn) matrix.
    ///
    /// The return value is: `(rows, cols, vector with data in column major mode)`
    ///
    /// # Example
    /// ```
    /// use automatica::{linear_system::SsGen, Continuous};
    /// let a = [-1., 3., 0., 2.];
    /// let b = [1., 2.];
    /// let c = [1., 1.];
    /// let d = [0.];
    /// let sys = SsGen::<_, Continuous>::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    /// let mr = sys.osservability();
    /// assert_eq!((2, 2, vec![1., 1., -1., 5.]), mr);
    /// ```
    pub fn osservability(&self) -> (usize, usize, Vec<T>) {
        let n = self.dim.states;
        let p = self.dim.outputs;
        let mo = observability_impl(n, p, &self.a, &self.c);
        (n, n * p, mo.data.as_vec().clone())
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

impl<T: Float + Scalar + ComplexField + RealField, U: Time> TryFrom<TfGen<T, U>> for SsGen<T, U> {
    type Error = &'static str;

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
    fn try_from(tf: TfGen<T, U>) -> Result<Self, Self::Error> {
        // Get the denominator in the monic form and the leading coefficient.
        let (den_monic, den_n) = tf.den().monic();
        // Extend the numerator coefficients with zeros to the length of the
        // denominator polynomial.
        let order = den_monic
            .degree()
            .expect("Transfer functions cannot have zero polynomial denominator");
        // Divide the denominator polynomial by the highest coefficient of the
        // numerator polinomial to mantain the original gain.
        let mut num = tf.num().clone() / den_n;
        num.extend(order);

        // Calculate the observability canonical form.
        let a = match den_monic.companion() {
            Some(a) => a,
            _ => return Err("Denominator has no poles"),
        };

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
        Ok(Self {
            a,
            b,
            c,
            d,
            dim: Dim {
                states,
                inputs: 1,
                outputs: 1,
            },
            time: PhantomData,
        })
    }
}

/// Implementation of state-space representation
impl<T: Scalar + Display, U: Time> Display for SsGen<T, U> {
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
    use crate::{polynomial::matrix::MatrixOfPoly, Continuous, Discrete};

    use nalgebra::DMatrix;

    #[quickcheck]
    fn dimensions(states: usize, inputs: usize, outputs: usize) -> bool {
        let d = Dim {
            states,
            inputs,
            outputs,
        };
        states == d.states() && inputs == d.inputs() && outputs == d.outputs()
    }

    #[test]
    fn system_dimensions() {
        let states = 2;
        let inputs = 1;
        let outputs = 1;
        let sys = SsGen::<_, Continuous>::new_from_slice(
            states,
            inputs,
            outputs,
            &[-2., 0., 3., -7.],
            &[1., 3.],
            &[-1., 0.5],
            &[0.1],
        );
        assert_eq!(
            Dim {
                states,
                inputs,
                outputs
            },
            sys.dim()
        );
    }

    #[test]
    #[should_panic]
    fn poles_fail() {
        let eig1 = -2.;
        let eig2 = -7.;
        let a = DMatrix::from_row_slice(2, 2, &[eig1, 0., 3., eig2]);
        let schur = Schur::new(a);
        //dbg!(&schur);
        let poles = schur.complex_eigenvalues();
        //dbg!(poles);
        assert_eq!((eig1, eig2), (poles[0].re, poles[1].re));
    }

    #[test]
    fn poles_regression() {
        let eig1 = -2.;
        let eig2 = -7.;
        let sys = SsGen::<_, Discrete>::new_from_slice(
            2,
            1,
            1,
            &[eig1, 0., 3., eig2],
            &[1., 3.],
            &[-1., 0.5],
            &[0.1],
        );
        let poles = sys.poles();
        assert_eq!((eig1, eig2), (poles[0].re, poles[1].re));
    }

    #[quickcheck]
    fn poles_two(eig1: f64, eig2: f64) -> bool {
        let sys = SsGen::<_, Continuous>::new_from_slice(
            2,
            1,
            1,
            &[eig1, 0., 3., eig2],
            &[1., 3.],
            &[-1., 0.5],
            &[0.1],
        );
        let poles = sys.poles();

        let mut expected = [eig1, eig2];
        expected.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mut actual = [poles[0].re, poles[1].re];
        actual.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        relative_eq!(expected[0], actual[0], max_relative = 1e-10)
            && relative_eq!(expected[1], actual[1], max_relative = 1e-10)
    }

    #[test]
    fn poles_three() {
        let eig1 = -7.;
        let eig2 = -2.;
        let eig3 = 1.25;
        let sys = SsGen::<_, Discrete>::new_from_slice(
            3,
            1,
            1,
            &[eig1, 0., 0., 3., eig2, 0., 10., 0.8, eig3],
            &[1., 3., -5.5],
            &[-1., 0.5, -4.3],
            &[0.],
        );
        let mut poles = sys.poles();
        poles.sort_unstable_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        assert_relative_eq!(eig1, poles[0].re, max_relative = 1e-10);
        assert_relative_eq!(eig2, poles[1].re, max_relative = 1e-10);
        assert_relative_eq!(eig3, poles[2].re, max_relative = 1e-10);
    }

    #[test]
    fn equilibrium() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1.];
        let d = [0.];

        let sys = SsGen::<_, Continuous>::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let u = 0.0;
        let eq = sys.equilibrium(&[u]).unwrap();
        assert_eq!((0., 0.), (eq.x()[0], eq.y()[0]));
        println!("{}", &eq);
        assert!(!format!("{}", eq).is_empty());
    }

    #[test]
    fn leverrier_algorythm() {
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
    fn leverrier_1x1_matrix() {
        let t = DMatrix::from_row_slice(1, 1, &[3.]);
        let expected_pc = Poly::new_from_coeffs(&[-3., 1.]);
        let expected_degree0 = DMatrix::from_row_slice(1, 1, &[1.]);

        let (p, poly_matrix) = leverrier(&t);
        assert_eq!(expected_pc, p);
        assert_eq!(expected_degree0, poly_matrix[0]);

        let mp = MatrixOfPoly::from(poly_matrix);
        let expected_result = "[[1]]";
        assert_eq!(expected_result, format!("{}", &mp));
    }

    #[test]
    fn convert_to_ss_continuous() {
        use crate::transfer_function::continuous::Tf;
        let tf = Tf::new(
            Poly::new_from_coeffs(&[1.]),
            Poly::new_from_coeffs(&[1., 1., 1.]),
        );

        let ss = SsGen::try_from(tf).unwrap();

        assert_eq!(DMatrix::from_row_slice(2, 2, &[0., -1., 1., -1.]), *ss.a());
        assert_eq!(DMatrix::from_row_slice(2, 1, &[1., 0.]), *ss.b());
        assert_eq!(DMatrix::from_row_slice(1, 2, &[0., 1.]), *ss.c());
        assert_eq!(DMatrix::from_row_slice(1, 1, &[0.]), *ss.d());
    }

    #[test]
    fn controllability() {
        let a = [-1., 3., 0., 2.];
        let b = [1., 2.];
        let c = [1., 1.];
        let d = [0.];

        let sys = SsGen::<_, Discrete>::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let mr = sys.controllability();
        assert_eq!((2, 2, vec![1., 2., 5., 4.]), mr);
    }

    #[test]
    fn osservability() {
        let a = [-1., 3., 0., 2.];
        let b = [1., 2.];
        let c = [1., 1.];
        let d = [0.];

        let sys = SsGen::<_, Continuous>::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let mo = sys.osservability();
        assert_eq!((2, 2, vec![1., 1., -1., 5.]), mo);
    }
}
