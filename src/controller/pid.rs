//! # PID (Proportional-integral-derivative) controller
//!
//! # Example
//! ```
//! use automatica::controller::pid::Pid;
//! let pid = Pid::new_ideal(10., 5., 2.);
//! let transfer_function = pid.tf();
//! ```

use crate::{polynomial::Poly, transfer_function::Tf};

use num_traits::Float;

/// Proportional-Integral-Derivative controller
pub struct Pid<F: Float> {
    /// Proportional action coefficient
    kp: F,
    /// Integral time
    ti: F,
    /// Derivative time
    td: F,
    /// Constant for additional pole
    n: Option<F>,
}

/// Implementation of Pid methods
impl<F: Float> Pid<F> {
    /// Create a new ideal PID controller
    ///
    /// # Arguments
    ///
    /// * `kp` - Proportional action coefficient
    /// * `ti` - Integral time
    /// * `td` - Derivative time
    pub fn new_ideal(kp: F, ti: F, td: F) -> Self {
        Self {
            kp,
            ti,
            td,
            n: None,
        }
    }

    /// Create a new real PID controller
    ///
    /// # Arguments
    ///
    /// * `kp` - Proportional action coefficient
    /// * `ti` - Integral time
    /// * `td` - Derivative time
    /// * `n` - Constant for additional pole
    pub fn new(kp: F, ti: F, td: F, n: F) -> Self {
        Self {
            kp,
            ti,
            td,
            n: Some(n),
        }
    }

    /// Calculate the transfer function of the PID controller
    ///
    /// # Real PID
    /// ```text
    ///          1         Td
    /// Kp (1 + ---- + ---------- s) =
    ///         Ti*s   1 + Td/N*s
    ///
    ///      N + (Ti*N +Td)s + Ti*Td(1 + N)s^2
    /// = Kp ----------------------------------
    ///             Ti*N*s + Ti*Td*s^2
    /// ```
    /// # Ideal PID
    ///
    /// ```text
    ///    1 + Ti*s + Ti*Ti*s^2
    /// Kp --------------------
    ///           Ti*s
    /// ```
    pub fn tf(&self) -> Tf<f64> {
        if let Some(n) = self.n {
            let a0 = self.kp * n;
            let a1 = self.kp * (self.ti * n + self.td);
            let a2 = self.kp * self.ti * self.td * (F::one() + n);
            let b0 = 0.;
            let b1 = self.ti * n;
            let b2 = self.ti * self.td;
            Tf::new(
                Poly::new_from_coeffs(&[
                    a0.to_f64().unwrap(),
                    a1.to_f64().unwrap(),
                    a2.to_f64().unwrap(),
                ]),
                Poly::new_from_coeffs(&[b0, b1.to_f64().unwrap(), b2.to_f64().unwrap()]),
            )
        } else {
            let ti = self.ti.to_f64().unwrap();
            let kp = self.kp.to_f64().unwrap();
            Tf::new(
                Poly::new_from_coeffs(&[1., kp * ti, kp * ti * self.td.to_f64().unwrap()]),
                Poly::new_from_coeffs(&[0., ti]),
            )
        }
    }
}

#[cfg(test)]
mod pid_tests {
    use super::*;
    use crate::Eval;
    use num_complex::Complex64;

    #[test]
    fn pid_creation() {
        let pid = Pid::new_ideal(10., 5., 2.);
        let tf = pid.tf();
        let c = tf.eval(&Complex64::new(0., 1.));
        assert_eq!(Complex64::new(10., 19.8), c);
    }
}
