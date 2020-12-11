//! # Continuous time linear system
//!
//! The time evolution of the system is performed through ODE (ordinary
//! differential equation) [solvers](../solver/index.html).

use nalgebra::{ComplexField, DVector, RealField};
use num_traits::Float;

use crate::{
    enums::Continuous,
    linear_system::{
        solver::{Order, Radau, Rk, Rkf45},
        Equilibrium, SsGen,
    },
    units::Seconds,
};

/// State-space representation of continuous time linear system
pub type Ss<T> = SsGen<T, Continuous>;

/// Implementation of the methods for the state-space
impl<T: ComplexField> Ss<T> {
    /// Calculate the equilibrium point for continuous time systems,
    /// given the input condition
    /// ```text
    /// x = - A^-1 * B * u
    /// y = - (C * A^-1 * B + D) * u
    /// ```
    ///
    /// # Arguments
    ///
    /// * `u` - Input vector
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ss;
    /// let a = [-1., 1., -1., 0.25];
    /// let b = [1., 0.25];
    /// let c = [0., 1., -1., 1.];
    /// let d = [0., 1.];
    ///
    /// let sys = Ss::new_from_slice(2, 1, 2, &a, &b, &c, &d);
    /// let u = 0.0;
    /// let eq = sys.equilibrium(&[u]).unwrap();
    /// assert_eq!((0., 0.), (eq.x()[0], eq.y()[0]));
    /// ```
    pub fn equilibrium(&self, u: &[T]) -> Option<Equilibrium<T>> {
        assert_eq!(u.len(), self.b.ncols(), "Wrong number of inputs.");
        let u = DVector::from_row_slice(u);
        // 0 = A*x + B*u
        let bu = -&self.b * &u;
        let lu = &self.a.clone().lu();
        // A*x = -B*u
        let x = lu.solve(&bu)?;
        // y = C*x + D*u
        let y = &self.c * &x + &self.d * u;
        Some(Equilibrium::new(x, y))
    }
}

/// Implementation of the methods for the state-space
impl<T: ComplexField + Float + RealField> Ss<T> {
    /// System stability. Checks if all A matrix eigenvalues (poles) are negative.
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ss;
    /// let sys = Ss::new_from_slice(2, 1, 1, &[-2., 0., 3., -7.], &[1., 3.], &[-1., 0.5], &[0.1]);
    /// assert!(sys.is_stable());
    /// ```
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|p| p.re.is_negative())
    }
}

/// Implementation of the methods for the state-space
impl Ss<f64> {
    /// Time evolution for the given input, using Runge-Kutta second order method
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column mayor)
    /// * `x0` - initial state (column mayor)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub fn rk2<F>(&self, u: F, x0: &[f64], h: Seconds<f64>, n: usize) -> Rk<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        Rk::new(self, u, x0, h, n, Order::Rk2)
    }

    /// Time evolution for the given input, using Runge-Kutta fourth order method
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column mayor)
    /// * `x0` - initial state (column mayor)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub fn rk4<F>(&self, u: F, x0: &[f64], h: Seconds<f64>, n: usize) -> Rk<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        Rk::new(self, u, x0, h, n, Order::Rk4)
    }

    /// Runge-Kutta-Fehlberg 45 with adaptive step for time evolution.
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `limit` - time evaluation limit
    /// * `tol` - error tolerance
    pub fn rkf45<F>(
        &self,
        u: F,
        x0: &[f64],
        h: Seconds<f64>,
        limit: Seconds<f64>,
        tol: f64,
    ) -> Rkf45<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        Rkf45::new(self, u, x0, h, limit, tol)
    }

    /// Radau of order 3 with 2 steps method for time evolution.
    ///
    /// # Arguments
    ///
    /// * `u` - input function returning a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    /// * `tol` - error tolerance
    pub fn radau<F>(&self, u: F, x0: &[f64], h: Seconds<f64>, n: usize, tol: f64) -> Radau<F, f64>
    where
        F: Fn(Seconds<f64>) -> Vec<f64>,
    {
        Radau::new(self, u, x0, h, n, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::many_single_char_names)]
    #[test]
    fn equilibrium() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1., -1., 1.];
        let d = [0., 1.];

        let sys = Ss::new_from_slice(2, 1, 2, &a, &b, &c, &d);
        let u = 0.0;
        let eq = sys.equilibrium(&[u]).unwrap();
        assert_eq!((0., 0.), (eq.x()[0], eq.y()[0]));
        assert!(!format!("{}", eq).is_empty());
    }

    #[test]
    fn stability() {
        let eig1 = -2.;
        let eig2 = -7.;
        let sys = Ss::new_from_slice(
            2,
            1,
            1,
            &[eig1, 0., 3., eig2],
            &[1., 3.],
            &[-1., 0.5],
            &[0.1],
        );
        assert!(sys.is_stable())
    }

    #[test]
    fn new_rk2() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1.];
        let d = [0.];
        let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let iter = sys.rk2(|_| vec![1.], &[0., 0.], Seconds(0.1), 30);
        assert_eq!(31, iter.count());
    }

    #[test]
    fn new_rk4() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1.];
        let d = [0.];
        let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let iter = sys.rk4(|_| vec![1.], &[0., 0.], Seconds(0.1), 30);
        assert_eq!(31, iter.count());
    }

    #[test]
    fn new_rkf45() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1.];
        let d = [0.];
        let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let iter = sys.rkf45(|_| vec![1.], &[0., 0.], Seconds(0.1), Seconds(2.), 1e-5);
        assert_relative_eq!(2., iter.last().unwrap().time().0, max_relative = 0.01);
    }

    #[test]
    fn new_radau() {
        let a = [-1., 1., -1., 0.25];
        let b = [1., 0.25];
        let c = [0., 1.];
        let d = [0.];
        let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
        let iter = sys.radau(|_| vec![1.], &[0., 0.], Seconds(0.1), 30, 1e-5);
        assert_eq!(31, iter.count());
    }
}
