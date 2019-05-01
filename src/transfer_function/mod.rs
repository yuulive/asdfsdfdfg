use crate::{polynomial::Poly, Eval};

use nalgebra::DVector;
use num_complex::Complex64;

use std::fmt;

/// Transfer function representation of a linear system
#[derive(Debug)]
pub struct Tf {
    /// Transfer function numerator
    num: Poly,
    /// Transfer function denominator
    den: Poly,
}

/// Implementation of transfer function methods
impl Tf {
    /// Create a new transfer function given its numerator and denominator
    ///
    /// # Arguments
    ///
    /// * `num` - Transfer function numerator
    /// * `den` - Transfer function denominator
    pub fn new(num: Poly, den: Poly) -> Self {
        Tf { num, den }
    }

    /// Extract transfer function numerator
    pub fn num(&self) -> Poly {
        self.num.clone()
    }

    /// Extract transfer function denominator
    pub fn den(&self) -> Poly {
        self.den.clone()
    }

    /// Calculate the poles of the transfer function
    pub fn poles(&self) -> DVector<f64> {
        self.den.roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn zeros(&self) -> DVector<f64> {
        self.num.roots()
    }
}

/// Implementation of the evaluation of a transfer function
impl Eval<Complex64> for Tf {
    fn eval(&self, s: Complex64) -> Complex64 {
        self.num.eval(s) / self.den.eval(s)
    }
}

/// Implementation of transfer function printing
impl fmt::Display for Tf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s_num = self.num.to_string();
        let s_den = self.den.to_string();

        let length = s_num.len().max(s_den.len());
        let dash = "─".repeat(length);

        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}
