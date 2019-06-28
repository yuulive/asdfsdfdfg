use crate::{
    linear_system::{self, Ss},
    polynomial::{Poly, MP},
    Eval,
};

use ndarray::{Array, Array2, Axis, Zip};
use num_complex::Complex64;

use std::convert::TryFrom;
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
    pub fn num(&self) -> &Poly {
        &self.num
    }

    /// Extract transfer function denominator
    pub fn den(&self) -> &Poly {
        &self.den
    }

    /// Calculate the poles of the transfer function
    pub fn poles(&self) -> Option<Vec<f64>> {
        self.den.roots()
    }

    /// Calculate the poles of the transfer function
    pub fn complex_poles(&self) -> Vec<Complex64> {
        self.den.complex_roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn zeros(&self) -> Option<Vec<f64>> {
        self.num.roots()
    }

    /// Calculate the zeros of the transfer function
    pub fn complex_zeros(&self) -> Vec<Complex64> {
        self.num.complex_roots()
    }
}

impl TryFrom<Ss> for Tf {
    type Error = &'static str;

    /// Convert a state-space representation into transfer functions.
    /// Conversion is available for Single Input Single Output system.
    /// If fails if the system is not SISO
    ///
    /// # Arguments
    ///
    /// `ss` - state space linear system
    fn try_from(ss: Ss) -> Result<Self, Self::Error> {
        let (pc, a_inv) = linear_system::leverrier(ss.a());
        let g = a_inv.left_mul(ss.c()).right_mul(ss.b());
        let rest = &pc * ss.d();
        let tf = g + rest;
        if let Some(num) = MP::from(tf).siso() {
            Ok(Self::new(num.clone(), pc))
        } else {
            Err("Linar system is not Single Input Single Output")
        }
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
    num: MP,
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
    pub fn new(num: MP, den: Poly) -> Self {
        Self { num, den }
    }
}

impl Eval<Vec<Complex64>> for TfMatrix {
    fn eval(&self, s: &Vec<Complex64>) -> Vec<Complex64> {
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
        // `.` means 'evaluated at'

        // Create a matrix to contain the result of the evalutation.
        let mut res = Array2::from_elem(self.num.matrix.dim(), Complex64::new(0.0, 0.0));

        // Evaluate the characteristic polynomial for each input returning a vector.
        let pc_vector = Array::from_vec(s.iter().map(|si| self.den.eval(si)).collect());

        // Zip the result and the numerator matrix row by row.
        Zip::from(res.genrows_mut())
            .and(self.num.matrix.genrows())
            .apply(|mut res_row, matrix_row| {
                // Zip the row of the result matrix.
                Zip::from(&mut res_row)
                    .and(s) // The vectror of the input.
                    .and(matrix_row) // The row of the numerator matrix.
                    .and(&pc_vector) // The characteristic polynomial.
                    .apply(|r, s, n, p| *r = n.eval(s) / p);
            });

        res.sum_axis(Axis(1)).to_vec()
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
        let rest = &pc * ss.d();
        let tf = g + rest;
        Self::new(MP::from(tf), pc)
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
