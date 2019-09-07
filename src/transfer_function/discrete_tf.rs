//! # Transfer function discretization
//!
//! The discretization can be performed with Euler or Tustin methods.

use crate::{linear_system::discrete::Discretization, transfer_function::Tf, Eval};

use num_complex::Complex64;

/// Discrete transfer function.
pub struct Tfz {
    /// Transfer function
    tf: Tf,
    /// Sampling period
    ts: f64,
    /// Discretization function
    conversion: fn(Complex64, f64) -> Complex64,
}

/// Implementation of `Tfz` struct.
impl Tfz {
    /// Create a new discrete transfer function from a continuous one.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `conversion` - Conversion function
    fn new_from_cont(tf: Tf, ts: f64, conversion: fn(Complex64, f64) -> Complex64) -> Self {
        Self { tf, ts, conversion }
    }

    /// Discretize a transfer function.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `method` - Discretization method
    pub fn discretize(tf: Tf, ts: f64, method: Discretization) -> Self {
        let conv = match method {
            Discretization::ForwardEuler => fe,
            Discretization::BackwardEuler => fb,
            Discretization::Tustin => tu,
        };
        Self::new_from_cont(tf, ts, conv)
    }
}

/// Forward Euler transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn fe(z: Complex64, ts: f64) -> Complex64 {
    (z - 1.) / ts
}

/// Backward Euler transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn fb(z: Complex64, ts: f64) -> Complex64 {
    (z - 1.) / (ts * z)
}

/// Tustin transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn tu(z: Complex64, ts: f64) -> Complex64 {
    2. / ts * (z - 1.) / (z + 1.)
}

/// Implementation of the evaluation of a transfer function
impl Eval<Complex64> for Tfz {
    fn eval(&self, z: &Complex64) -> Complex64 {
        let s = (self.conversion)(*z, self.ts);
        self.tf.eval(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::Poly;
    use crate::{Decibel, Eval};
    use num_complex::Complex64;

    #[test]
    fn new_tfz() {
        let _t = Tfz {
            tf: Tf::new(
                Poly::new_from_coeffs(&[0., 1.]),
                Poly::new_from_coeffs(&[1., 1., 1.]),
            ),
            ts: 0.1,
            conversion: |_z, _ts| 1.0 * Complex64::i(),
        };
    }

    #[test]
    fn eval() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[0.2, -0.4]),
            Poly::new_from_coeffs(&[0., 1., 0.2, 0.01]),
        );
        let tfz = Tfz::discretize(tf, 1., Discretization::BackwardEuler);
        let z = 0.5 * Complex64::i();
        let s = (tfz.conversion)(z, 1.);
        assert_eq!(Complex64::new(1., 2.), s);

        let g = tfz.eval(&z);
        assert_eq!(
            (-10.602811176458955, 171.91911477161943),
            (g.norm().to_db(), g.arg().to_degrees())
        );
    }
}
