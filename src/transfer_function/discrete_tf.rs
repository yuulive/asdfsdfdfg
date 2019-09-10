//! # Transfer function discretization
//!
//! The discretization can be performed with Euler or Tustin methods.

use crate::{linear_system::discrete::Discretization, transfer_function::Tf, units::Seconds, Eval};

use num_complex::Complex64;

/// Discrete transfer function.
pub struct Tfz {
    /// Transfer function
    tf: Tf,
    /// Sampling period
    ts: Seconds,
    /// Discretization function
    conversion: fn(Complex64, Seconds) -> Complex64,
}

/// Implementation of `Tfz` struct.
impl Tfz {
    /// Create a new discrete transfer function from a continuous one.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `conversion` - Conversion function
    fn new_from_cont(tf: Tf, ts: Seconds, conversion: fn(Complex64, Seconds) -> Complex64) -> Self {
        Self { tf, ts, conversion }
    }

    /// Discretize a transfer function.
    ///
    /// # Arguments
    /// * `tf` - Continuous transfer function
    /// * `ts` - Sampling period
    /// * `method` - Discretization method
    pub fn discretize(tf: Tf, ts: Seconds, method: Discretization) -> Self {
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
fn fe(z: Complex64, ts: Seconds) -> Complex64 {
    (z - 1.) / ts.0
}

/// Backward Euler transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn fb(z: Complex64, ts: Seconds) -> Complex64 {
    (z - 1.) / (ts.0 * z)
}

/// Tustin transformation
///
/// # Arguments
/// * `z` - Discrete evaluation point
/// * `ts` - Sampling period
fn tu(z: Complex64, ts: Seconds) -> Complex64 {
    2. / ts.0 * (z - 1.) / (z + 1.)
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
    use crate::{units::Decibel, Eval};
    use num_complex::Complex64;

    #[test]
    fn new_tfz() {
        let _t = Tfz {
            tf: Tf::new(
                Poly::new_from_coeffs(&[0., 1.]),
                Poly::new_from_coeffs(&[1., 1., 1.]),
            ),
            ts: Seconds(0.1),
            conversion: |_z, _ts| 1.0 * Complex64::i(),
        };
    }

    #[test]
    fn eval() {
        let tf = Tf::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let g = tf.eval(&z);
        assert_relative_eq!(20.159, g.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(75.828, g.arg().to_degrees(), max_relative = 1e-4);

        let ts = Seconds(1.);
        let tfz = Tfz::discretize(tf, ts, Discretization::Tustin);
        let s = (tfz.conversion)(z, ts);
        assert_eq!(Complex64::new(-1.2, 1.6), s);

        let gz = tfz.eval(&z);
        assert_relative_eq!(32.753, gz.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(114.20, gz.arg().to_degrees(), max_relative = 1e-4);
    }
}
