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
pub struct Pid<T: Float> {
    /// Proportional action coefficient
    kp: T,
    /// Integral time
    ti: T,
    /// Derivative time
    td: T,
    /// Constant for additional pole
    n: Option<T>,
}

/// Implementation of Pid methods
impl<T: Float> Pid<T> {
    /// Create a new ideal PID controller
    ///
    /// # Arguments
    ///
    /// * `kp` - Proportional action coefficient
    /// * `ti` - Integral time
    /// * `td` - Derivative time
    pub fn new_ideal(kp: T, ti: T, td: T) -> Self {
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
    pub fn new(kp: T, ti: T, td: T, n: T) -> Self {
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
    pub fn tf(&self) -> Tf<T> {
        if let Some(n) = self.n {
            let a0 = self.kp * n;
            let a1 = self.kp * (self.ti * n + self.td);
            let a2 = self.kp * self.ti * self.td * (T::one() + n);
            let b0 = T::zero();
            let b1 = self.ti * n;
            let b2 = self.ti * self.td;
            Tf::new(
                Poly::new_from_coeffs(&[a0, a1, a2]),
                Poly::new_from_coeffs(&[b0, b1, b2]),
            )
        } else {
            Tf::new(
                Poly::new_from_coeffs(&[T::one(), self.ti, self.ti * self.td]),
                Poly::new_from_coeffs(&[T::zero(), self.ti / self.kp]),
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
        assert_eq!(Complex64::new(10., 18.), c);
    }
}
