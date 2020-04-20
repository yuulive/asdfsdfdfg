//! # PID (Proportional-integral-derivative) controller
//!
//! Common industrial controllers.
//! * real PID
//! * ideal PID
//! * automatic calculation of the corrisponding transfer function

use crate::{polynomial::Poly, transfer_function::continuous::Tf};

use num_traits::Float;

/// Proportional-Integral-Derivative controller
#[derive(Debug)]
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
    ///
    /// # Example
    /// ```
    /// use automatica::controller::pid::Pid;
    /// let pid = Pid::new_ideal(4., 6., 0.1);
    /// ```
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
    ///
    /// # Example
    /// ```
    /// use automatica::controller::pid::Pid;
    /// let pid = Pid::new(4., 6., 12., 0.1);
    /// ```
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
    ///    1 + Ti*s + Ti*Td*s^2
    /// Kp --------------------
    ///           Ti*s
    /// ```
    ///
    /// # Example
    /// ```
    /// #[macro_use] extern crate automatica;
    /// use automatica::{controller::pid::Pid, Tf};
    /// let pid = Pid::new_ideal(2., 2., 0.5);
    /// let tf = Tf::new(poly![1., 2., 1.], poly![0., 1.]);
    /// assert_eq!(tf, pid.tf());
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
    use crate::{units::Decibel, Eval};
    use num_complex::Complex64;

    #[test]
    fn ideal_pid_creation() {
        let pid = Pid::new_ideal(10., 5., 2.);
        let tf = pid.tf();
        let c = tf.eval(Complex64::new(0., 1.));
        assert_eq!(Complex64::new(10., 18.), c);
    }

    #[test]
    fn real_pid_creation() {
        // Example 15.1
        let g = Tf::new(
            Poly::new_from_coeffs(&[1.]),
            Poly::new_from_roots(&[-1., -1., -1.]),
        );
        let pid = Pid::new(2., 2., 0.5, 5.);
        let r = pid.tf();
        assert_eq!(Some(vec![-10., 0.]), r.real_poles());
        let l = &g * &r;
        let critical_freq = 0.8;
        let c = l.eval(Complex64::new(0., critical_freq));
        assert_abs_diff_eq!(0., c.norm().to_db(), epsilon = 0.1);
    }
}
