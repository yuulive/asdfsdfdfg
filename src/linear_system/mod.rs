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
        let inv_a = &self.a.clone().try_inverse().unwrap();
        let x = inv_a * &self.b * &u;
        let y = (-&self.c * inv_a * &self.b + &self.d) * u;
        Some(Equilibrium::new(x, y))
    }
}

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

impl fmt::Display for Equilibrium {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}\ny: {}", self.x, self.y)
    }
}
