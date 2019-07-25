use crate::{
    polynomial::{Poly, PolyMatrix},
    transfer_function::Tf,
};

use nalgebra::{DMatrix, DVector, Schur};
use num_complex::Complex64;

use std::convert::From;
use std::fmt;

/// State-space representation of a linar system
#[derive(Debug)]
pub struct Ss {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
}

/// Implementation of the methods for the state-space
impl Ss {
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
        a: &[f64],
        b: &[f64],
        c: &[f64],
        d: &[f64],
    ) -> Self {
        Self {
            a: DMatrix::from_row_slice(states, states, a),
            b: DMatrix::from_row_slice(states, inputs, b),
            c: DMatrix::from_row_slice(outputs, states, c),
            d: DMatrix::from_row_slice(outputs, inputs, d),
        }
    }

    /// Get the A matrix
    pub(crate) fn a(&self) -> &DMatrix<f64> {
        &self.a
    }

    /// Get the C matrix
    pub(crate) fn b(&self) -> &DMatrix<f64> {
        &self.b
    }

    /// Get the C matrix
    pub(crate) fn c(&self) -> &DMatrix<f64> {
        &self.c
    }

    /// Get the D matrix
    pub(crate) fn d(&self) -> &DMatrix<f64> {
        &self.d
    }

    /// Calculate the poles of the system
    pub fn poles(&self) -> Vec<Complex64> {
        Schur::new(self.a.clone())
            .complex_eigenvalues()
            .as_slice()
            .to_vec()
    }

    /// Calculate the equilibrium point for the given input condition
    ///
    /// # Arguments
    ///
    /// * `u` - Input vector
    pub fn equilibrium(&self, u: &[f64]) -> Option<Equilibrium> {
        assert_eq!(u.len(), self.b.ncols(), "Wrong number of inputs.");
        let u = DVector::from_row_slice(u);
        let inv_a = &self.a.clone().try_inverse()?;
        let x = -inv_a * &self.b * &u;
        let y = (-&self.c * inv_a * &self.b + &self.d) * u;
        Some(Equilibrium::new(x, y))
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
#[allow(non_snake_case)]
pub(crate) fn leverrier(A: &DMatrix<f64>) -> (Poly, PolyMatrix) {
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
        ak = -f64::from(k as u32).recip() * ABk.trace();
        a.insert(0, ak);
    }
    (Poly::new_from_coeffs(&a), PolyMatrix::new_from_coeffs(&B))
}

impl From<Tf> for Ss {
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
    ///
    /// # Arguments
    ///
    /// `tf` - transfer function
    fn from(tf: Tf) -> Self {
        // Extend the numerator coefficients with zeros to the length of the
        // denominator polynomial.
        let den = tf.den();
        let order = den.degree();
        let mut num = tf.num().clone();
        num.extend(order);

        // Calculate the observability canonical form.
        let a = den.companion();

        // Get the number of states n.
        let states = a.nrows();
        // Get the highest coefficient of the numerator.
        let b_n = num[order];

        // Create a nx1 vector with b'i = bi - ai * b'n
        let b = DMatrix::from_fn(states, 1, |i, _j| num[i] - den[i] * b_n);

        // Crate a 1xn vector with all zeros but the last that is 1.
        let mut c = DMatrix::zeros(1, states);
        c[states - 1] = 1.0;

        // Crate a 1x1 matrix with the highest coefficient of the numerator.
        let d = DMatrix::from_element(1, 1, b_n);

        Self { a, b, c, d }
    }
}

/// Implementation of state-space representation
impl fmt::Display for Ss {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "A: {}\nB: {}\nC: {}\nD: {}",
            self.a, self.b, self.c, self.d
        )
    }
}

/// Struct describing an equilibrium point
#[derive(Debug)]
pub struct Equilibrium {
    /// State equilibrium
    x: DVector<f64>,
    /// Output equilibrium
    y: DVector<f64>,
}

/// Implement methods for equilibrium
impl Equilibrium {
    /// Create a new equilibrium given the state and the output vectors
    ///
    /// # Arguments
    ///
    /// * `x` - State equilibrium
    /// * `y` - Output equilibrium
    pub(crate) fn new(x: DVector<f64>, y: DVector<f64>) -> Self {
        Equilibrium { x, y }
    }

    /// Retreive state coordinates for equilibrium
    pub fn x(&self) -> &[f64] {
        self.x.as_slice()
    }

    /// Retreive output coordinates for equilibrium
    pub fn y(&self) -> &[f64] {
        self.y.as_slice()
    }
}

/// Implementation of printing of equilibrium point
impl fmt::Display for Equilibrium {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}\ny: {}", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_leverrier() {
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
    fn convert_to_ss_1_test() {
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
    fn convert_to_ss_2_test() {
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
