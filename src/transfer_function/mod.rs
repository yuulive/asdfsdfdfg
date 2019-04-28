use crate::polynomial::{Eval, Poly};

use nalgebra::DVector;
use num_complex::Complex64;

pub struct Tf {
    num: Poly,
    den: Poly,
}

impl Tf {
    pub fn new(num: Poly, den: Poly) -> Self {
        Tf { num, den }
    }

    pub fn num(&self) -> Poly {
        self.num.clone()
    }

    pub fn den(&self) -> Poly {
        self.den.clone()
    }

    pub fn poles(&self) -> DVector<f64> {
        self.den.roots()
    }

    pub fn zeros(&self) -> DVector<f64> {
        self.num.roots()
    }
}

impl Eval<Complex64> for Tf {
    fn eval(&self, s: Complex64) -> Complex64 {
        self.num.eval(s) / self.den.eval(s)
    }
}
