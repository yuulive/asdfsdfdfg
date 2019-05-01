use crate::polynomial::Poly;

use nalgebra::{DMatrix, DVector, Schur};
use num_complex::Complex64;

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
    /// * `a` - A matrix
    /// * `b` - B matrix
    /// * `c` - C matrix
    /// * `d` - D matrix
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        Ss { a, b, c, d }
    }

    /// Get the A matrix
    pub fn a(&self) -> DMatrix<f64> {
        self.a.clone()
    }

    /// Get the C matrix
    pub fn b(&self) -> DMatrix<f64> {
        self.b.clone()
    }

    /// Get the C matrix
    pub fn c(&self) -> DMatrix<f64> {
        self.c.clone()
    }

    /// Get the D matrix
    pub fn d(&self) -> DMatrix<f64> {
        self.d.clone()
    }

    /// Calculate the poles of the system
    pub fn poles(&self) -> DVector<Complex64> {
        Schur::new(self.a.clone()).complex_eigenvalues()
    }

    /// Calculate the equilibrium point for the given input condition
    ///
    /// # Arguments
    ///
    /// * `u` - Input vector
    pub fn equilibrium(&self, u: &[f64]) -> Option<Equilibrium> {
        let u = DMatrix::from_row_slice(u.len(), 1, u);
        let inv_a = &self.a.clone().try_inverse()?;
        let x = inv_a * &self.b * &u;
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
/// Bk = a_(k-1)I* + A*B_(k-1)
#[allow(non_snake_case)]
pub fn leverrier(A: &DMatrix<f64>) -> (Poly, Vec<DMatrix<f64>>) {
    let size = A.nrows(); // A is a square matrix.
    let mut a = vec![1.0];
    let a1 = -A.trace();
    a.insert(0, a1);

    let B1 = DMatrix::identity(size, size); // eye(n,n)
    let mut B = vec![B1.clone()];
    if size == 1 {
        return (Poly::new_from_coeffs(&a), B);
    }

    let mut Bk = B1.clone();
    let mut ak = a1;
    for k in 2..=size {
        Bk = ak * &B1 + A * &Bk;
        B.insert(0, Bk.clone());

        let ABk = A * &Bk;
        ak = -f64::from(k as u32).recip() * &ABk.trace();
        a.insert(0, ak);
    }
    (Poly::new_from_coeffs(&a), B)
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
    x: DMatrix<f64>,
    /// Output equilibrium
    y: DMatrix<f64>,
}

/// Implement methods for equilibrium
impl Equilibrium {
    /// Create a new equilibrium given the state and the output vectors
    ///
    /// # Arguments
    ///
    /// * `x` - State equilibrium
    /// * `y` - Output equilibrium
    pub fn new(x: DMatrix<f64>, y: DMatrix<f64>) -> Self {
        Equilibrium { x, y }
    }
}

/// Implementation of printing of equilibrium point
impl fmt::Display for Equilibrium {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}\ny: {}", self.x, self.y)
    }
}
