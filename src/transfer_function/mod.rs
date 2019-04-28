use crate::polynomial::Poly;

use nalgebra::DVector;

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
