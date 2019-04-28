use nalgebra::{DMatrix, DVector, Schur};
use num_complex::Complex64;

use std::fmt;

#[derive(Debug)]
pub struct Ss {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
}

impl Ss {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        Ss { a, b, c, d }
    }

    pub fn a(&self) -> DMatrix<f64> {
        self.a.clone()
    }
    pub fn b(&self) -> DMatrix<f64> {
        self.b.clone()
    }
    pub fn c(&self) -> DMatrix<f64> {
        self.c.clone()
    }
    pub fn d(&self) -> DMatrix<f64> {
        self.d.clone()
    }

    pub fn poles(&self) -> DVector<Complex64> {
        Schur::new(self.a.clone()).complex_eigenvalues()
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
