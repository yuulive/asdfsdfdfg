use crate::{
    linear_system::{self, Ss},
    polynomial::{Poly, PolyMatrix},
    Eval,
};

use nalgebra::{DMatrix, DVector};
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
    pub fn poles(&self) -> Option<DVector<f64>> {
        self.den.roots()
    }

    /// Calculate the poles of the transfer function
    pub fn complex_poles(&self) -> DVector<Complex64> {
        self.den.complex_roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn zeros(&self) -> Option<DVector<f64>> {
        self.num.roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn complex_zeros(&self) -> DVector<Complex64> {
        self.num.complex_roots()
    }
}

/// Implementation of the evaluation of a transfer function
impl Eval<Complex64> for Tf {
    fn eval(&self, s: &Complex64) -> Complex64 {
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

/// Matrix of transfer functions
pub struct TfMatrix {
    /// Polynomial matrix of the numerators
    num: PolyMatrix,
    /// Common polynomial denominator
    den: Poly,
}

/// Implementation of transfer function matrix
impl TfMatrix {
    /// Create a new transfer function matrix
    ///
    /// # Arguments
    ///
    /// * `num` - Polynomial matrix
    /// * `den` - Characteristic polynomial of the system
    pub fn new(num: PolyMatrix, den: Poly) -> Self {
        Self { num, den }
    }
}

impl Eval<DVector<Complex64>> for TfMatrix {
    fn eval(&self, s: &DVector<Complex64>) -> DVector<Complex64> {
        //
        // ┌  ┐ ┌┌         ┐ ┌     ┐┐┌  ┐
        // │y1│=││1/pc 1/pc│*│n1 n2│││s1│
        // │y2│ ││1/pc 1/pc│ │n3 n4│││s2│
        // └  ┘ └└         ┘ └     ┘┘└  ┘
        // `*` is the element by element multiplication
        // ┌     ┐ ┌┌         ┐ ┌     ┐┐ ┌┌     ┐ ┌     ┐┐
        // │y1 y2│=││1/pc 1/pc│.│s1 s2││*││n1 n2│.│s1 s2││
        // │y3 y4│ ││1/pc 1/pc│ │s1 s2││ ││n3 n4│ │s1 s2││
        // └     ┘ └└         ┘ └     ┘┘ └└     ┘ └     ┘┘
        // `.` means 'evaluated in'
        let rows = self.num[0].nrows();
        let cols = self.num[0].ncols();

        let mut s_matr = DMatrix::zeros(rows, cols);
        for r in 0..rows {
            s_matr.set_row(r, &s.transpose());
        }

        let num_matr = self.num.eval(&s_matr);

        let pc_matr = s.map(|si| self.den.eval(&si)).transpose();
        let mut den_matr = DMatrix::zeros(rows, cols);
        for r in 0..rows {
            den_matr.set_row(r, &pc_matr);
        }

        num_matr.component_div(&den_matr).column_sum()
    }
}

impl From<Ss> for TfMatrix {
    /// Convert a state-space representation into a matrix of transfer functions
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn from(ss: Ss) -> Self {
        let (pc, a_inv) = linear_system::leverrier(ss.a());
        let g = a_inv.left_mul(ss.c()).right_mul(ss.b());
        let rest = pc.clone() * ss.d().clone();
        let tf = g + rest;
        Self::new(tf, pc.clone())
    }
}

/// Implementation of transfer function matrix printing
impl fmt::Display for TfMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s_num = self.num.to_string();
        let s_den = self.den.to_string();

        let length = s_den.len();
        let dash = "─".repeat(length);

        write!(f, "{}\n{}\n{}", s_num, dash, s_den)
    }
}
