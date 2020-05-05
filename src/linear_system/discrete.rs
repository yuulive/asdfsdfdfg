//! # Discrete linear system
//!
//! Time evolution of the system is performed through successive
//! matrix multiplications.
//!
//! This module contains the algorithm for the discretization of
//! [continuous](../continuous/index.html) systems:
//! * forward Euler method
//! * backward Euler method
//! * Tustin (trapezoidal) method

use nalgebra::{ComplexField, DMatrix, DVector, RealField, Scalar};
use num_traits::Float;

use std::{
    marker::PhantomData,
    ops::{AddAssign, MulAssign},
};

use crate::{
    linear_system::{continuous::Ss, Equilibrium, SsGen},
    Discrete, Discretization,
};

/// State-space representation of discrete time linear system
pub type Ssd<T> = SsGen<T, Discrete>;

/// Implementation of the methods for the state-space
impl<T: ComplexField> Ssd<T> {
    /// Calculate the equilibrium point for discrete time systems,
    /// given the input condition.
    /// Input vector must have the same number of inputs of the system.
    /// ```text
    /// x = (I-A)^-1 * B * u
    /// y = (C * (I-A)^-1 * B + D) * u
    /// ```
    ///
    /// # Arguments
    ///
    /// * `u` - Input vector
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ssd;
    /// let a = [-1., 1., -1., 0.25];
    /// let b = [1., 0.25];
    /// let c = [0., 1., -1., 1.];
    /// let d = [0., 1.];
    ///
    /// let sys = Ssd::new_from_slice(2, 1, 2, &a, &b, &c, &d);
    /// let u = 0.0;
    /// let eq = sys.equilibrium(&[u]).unwrap();
    /// assert_eq!((0., 0.), (eq.x()[0], eq.y()[0]));
    /// ```
    pub fn equilibrium(&self, u: &[T]) -> Option<Equilibrium<T>> {
        if u.len() != self.dim.inputs() {
            eprintln!("Wrong number of inputs.");
            return None;
        }
        let u = DVector::from_row_slice(u);
        // x = A*x + B*u -> (I-A)*x = B*u
        let bu = &self.b * &u;
        let lu = (DMatrix::identity(self.a.nrows(), self.a.ncols()) - &self.a.clone()).lu();
        // (I-A)*x = -B*u
        let x = lu.solve(&bu)?;
        // y = C*x + D*u
        let y = &self.c * &x + &self.d * u;
        Some(Equilibrium::new(x, y))
    }
}

/// Trait for the set of methods on discrete linear systems.
impl<T: Scalar> Ssd<T> {
    /// Time evolution for a discrete linear system.
    ///
    /// # Arguments
    ///
    /// * `step` - simulation length
    /// * `input` - input function
    /// * `x0` - initial state
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// use automatica::{linear_system::discrete::DiscreteTime, Discretization, Ssd};
    /// let disc_sys = Ssd::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
    /// let impulse = |t| if t == 0 { vec![1.] } else { vec![0.] };
    /// let evo = disc_sys.evolution_fn(20, impulse, &[0., 0.]);
    /// let last = evo.last().unwrap();
    /// assert_abs_diff_eq!(0., last.state()[1], epsilon = 0.001);
    /// ```
    pub fn evolution_fn<F>(&self, steps: usize, input: F, x0: &[T]) -> EvolutionFn<F, T>
    where
        F: Fn(usize) -> Vec<T>,
    {
        let state = DVector::from_column_slice(x0);
        let next_state = DVector::from_column_slice(x0);
        EvolutionFn {
            sys: &self,
            time: 0,
            steps,
            input,
            state,
            next_state,
        }
    }

    /// Time evolution for a discrete linear system.
    ///
    /// # Arguments
    ///
    /// * `iter` - input data
    /// * `x0` - initial state
    ///
    /// # Example
    /// ```
    /// use std::iter;
    /// use automatica::{linear_system::discrete::DiscreteTime, Discretization, Ssd};
    /// let disc_sys = Ssd::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
    /// let impulse = iter::once(vec![1.]).chain(iter::repeat(vec![0.])).take(20);
    /// let evo = disc_sys.evolution_iter(impulse, &[0., 0.]);
    /// let last = evo.last().unwrap();
    /// assert!(last[0] < 0.001);
    /// ```
    pub fn evolution_iter<I, II>(&self, iter: II, x0: &[T]) -> EvolutionIter<I, T>
    where
        II: IntoIterator<Item = Vec<T>, IntoIter = I>,
        I: Iterator<Item = Vec<T>>,
    {
        let state = DVector::from_column_slice(x0);
        let next_state = DVector::from_column_slice(x0);
        EvolutionIter {
            sys: &self,
            state,
            next_state,
            iter: iter.into_iter(),
        }
    }
}

impl<T: ComplexField + Float + RealField> Ssd<T> {
    /// System stability. Checks if all A matrix eigenvalues (poles) are inside
    /// the unit circle.
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::Ssd;
    /// let sys = Ssd::new_from_slice(2, 1, 1, &[-0.2, 0., 3., 0.1], &[1., 3.], &[-1., 0.5], &[0.1]);
    /// assert!(sys.is_stable());
    /// ```
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.poles().iter().all(|p| p.abs() < T::one())
    }
}

/// Trait for the discretization of continuous time linear systems.
pub trait DiscreteTime<T: Scalar> {
    /// Convert a linear system into a discrete system.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    /// * `method` - discretization method
    fn discretize(&self, st: T, method: Discretization) -> Option<Ssd<T>>;
}

impl<T: ComplexField + Float> DiscreteTime<T> for Ss<T> {
    /// Convert a linear system into a discrete system.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    /// * `method` - discretization method
    ///
    /// # Example
    /// ```
    /// # #[macro_use] extern crate approx;
    /// use automatica::{linear_system::discrete::DiscreteTime, Discretization, Ss};
    /// let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
    /// let disc_sys = sys.discretize(0.1, Discretization::Tustin).unwrap();
    /// let evo = disc_sys.evolution_fn(20, |t| vec![1.], &[0., 0.]);
    /// let last = evo.last().unwrap();
    /// assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    /// ```
    fn discretize(&self, st: T, method: Discretization) -> Option<Ssd<T>> {
        match method {
            Discretization::ForwardEuler => self.forward_euler(st),
            Discretization::BackwardEuler => self.backward_euler(st),
            Discretization::Tustin => self.tustin(st),
        }
    }
}

impl<T: ComplexField + Float> Ss<T> {
    /// Discretization using forward Euler Method.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    fn forward_euler(&self, st: T) -> Option<Ssd<T>> {
        let states = self.dim.states;
        let identity = DMatrix::identity(states, states);
        Some(Ssd {
            a: identity + &self.a * st,
            b: &self.b * st,
            c: self.c.clone(),
            d: self.d.clone(),
            dim: self.dim,
            time: PhantomData,
        })
    }

    /// Discretization using backward Euler Method.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    fn backward_euler(&self, st: T) -> Option<Ssd<T>> {
        let states = self.dim.states;
        let identity = DMatrix::identity(states, states);
        let a = (identity - &self.a * st).try_inverse()?;
        Some(Ssd {
            b: &a * &self.b * st,
            c: &self.c * &a,
            d: &self.d + &self.c * &a * &self.b * st,
            a,
            dim: self.dim,
            time: PhantomData,
        })
    }

    /// Discretization using Tustin Method.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    fn tustin(&self, st: T) -> Option<Ssd<T>> {
        let states = self.dim.states;
        let identity = DMatrix::identity(states, states);
        // Casting is safe for both f32 and f64, representation is exact.
        let n_05 = T::from(0.5_f32).unwrap();
        let a_05_st = &self.a * (n_05 * st);
        let k = (&identity - &a_05_st).try_inverse()?;
        let b = &k * &self.b * st;
        Some(Ssd {
            a: &k * (&identity + &a_05_st),
            c: &self.c * &k,
            d: &self.d + &self.c * &b * n_05,
            b,
            dim: self.dim,
            time: PhantomData,
        })
    }
}

/// Struct to hold the iterator for the evolution of the discrete linear system.
/// It uses function to supply inputs.
#[derive(Debug)]
pub struct EvolutionFn<'a, F, T>
where
    F: Fn(usize) -> Vec<T>,
    T: Scalar,
{
    sys: &'a Ssd<T>,
    time: usize,
    steps: usize,
    input: F,
    state: DVector<T>,
    next_state: DVector<T>,
}

impl<'a, F, T> Iterator for EvolutionFn<'a, F, T>
where
    F: Fn(usize) -> Vec<T>,
    T: AddAssign + Float + MulAssign + Scalar,
{
    type Item = TimeEvolution<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.time > self.steps {
            None
        } else {
            let current_time = self.time;
            let u = DVector::from_vec((self.input)(current_time));
            // Copy `next_state` of the previous iteration into
            // the current `state`.
            std::mem::swap(&mut self.state, &mut self.next_state);
            self.next_state = &self.sys.a * &self.state + &self.sys.b * &u;
            let output = &self.sys.c * &self.state + &self.sys.d * &u;
            self.time += 1;
            Some(TimeEvolution {
                time: current_time,
                state: self.state.as_slice().to_vec(),
                output: output.as_slice().to_vec(),
            })
        }
    }
}

/// Struct to hold the iterator for the evolution of the discrete linear system.
/// It uses iterators to supply inputs.
#[derive(Debug)]
pub struct EvolutionIter<'a, I, T>
where
    I: Iterator<Item = Vec<T>>,
    T: Scalar,
{
    sys: &'a Ssd<T>,
    state: DVector<T>,
    next_state: DVector<T>,
    iter: I,
}

impl<'a, I, T> Iterator for EvolutionIter<'a, I, T>
where
    I: Iterator<Item = Vec<T>>,
    T: AddAssign + Float + MulAssign + Scalar,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let u_vec = self.iter.next()?;
        let u = DVector::from_vec(u_vec);
        // Copy `next_state` of the previous iteration into
        // the current `state`.
        std::mem::swap(&mut self.state, &mut self.next_state);
        self.next_state = &self.sys.a * &self.state + &self.sys.b * &u;
        let output = &self.sys.c * &self.state + &self.sys.d * &u;
        Some(output.as_slice().to_vec())
    }
}

/// Struct to hold the result of the discrete linear system evolution.
#[derive(Debug)]
pub struct TimeEvolution<T> {
    time: usize,
    state: Vec<T>,
    output: Vec<T>,
}

impl<T> TimeEvolution<T> {
    /// Get the time of the current step
    #[must_use]
    pub fn time(&self) -> usize {
        self.time
    }

    /// Get the current state of the system
    #[must_use]
    pub fn state(&self) -> &Vec<T> {
        &self.state
    }

    /// Get the current output of the system
    #[must_use]
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::Poly;

    #[test]
    fn equilibrium() {
        let a = &[0., 0.8, 0.4, 1., 0., 0., 0., 1., 0.7];
        let b = &[0., 1., 0., 0., -1., 0.];
        let c = &[1., 1.8, 1.1];
        let d = &[-1., 1.];
        let u = &[170., 0.];

        let sys = Ssd::new_from_slice(3, 2, 1, a, b, c, d);
        let eq = sys.equilibrium(u).unwrap();

        assert_relative_eq!(200.0, eq.x()[0]);
        assert_relative_eq!(200.0, eq.x()[1]);
        assert_relative_eq!(100.0, eq.x()[2], max_relative = 1e-10);
        assert_relative_eq!(500.0, eq.y()[0]);

        // Test wrong number of inputs.
        assert!(sys.equilibrium(&[0., 0., 0.]).is_none());
    }

    #[test]
    fn stability() {
        let a = &[0., 0.8, 0.4, 1., 0., 0., 0., 1., 0.7];
        let b = &[0., 1., 0., 0., -1., 0.];
        let c = &[1., 1.8, 1.1];
        let d = &[-1., 1.];

        let sys = Ssd::new_from_slice(3, 2, 1, a, b, c, d);

        assert!(!sys.is_stable());
    }

    #[test]
    fn convert_to_ss_discrete() {
        use crate::transfer_function::discrete::Tfz;
        let tf = Tfz::new(
            Poly::new_from_coeffs(&[1., 0., 1.]),
            Poly::new_from_coeffs(&[3., 4., 1.]),
        );

        let ss = Ssd::new_observability_realization(&tf).unwrap();

        assert_eq!(DMatrix::from_row_slice(2, 2, &[0., -3., 1., -4.]), ss.a);
        assert_eq!(DMatrix::from_row_slice(2, 1, &[-2., -4.]), ss.b);
        assert_eq!(DMatrix::from_row_slice(1, 2, &[0., 1.]), ss.c);
        assert_eq!(DMatrix::from_row_slice(1, 1, &[1.]), ss.d);
    }

    #[test]
    fn time_evolution() {
        let disc_sys =
            Ssd::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
        let impulse = |t| if t == 0 { vec![1.] } else { vec![0.] };
        let evo = disc_sys.evolution_fn(20, impulse, &[0., 0.]);
        let last = evo.last().unwrap();
        assert_eq!(20, last.time());
        assert_abs_diff_eq!(0., last.state()[1], epsilon = 0.001);
        assert_abs_diff_eq!(0., last.output()[0], epsilon = 0.001);
    }

    #[test]
    fn time_evolution_iter() {
        use std::iter;
        let disc_sys =
            Ssd::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
        let impulse = iter::once(vec![1.]).chain(iter::repeat(vec![0.])).take(20);
        let evo = disc_sys.evolution_iter(impulse, &[0., 0.]);
        let last = evo.last().unwrap();
        assert!(last[0] < 0.001);
    }

    #[test]
    fn discretization_tustin() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::Tustin).unwrap();
        let evo = disc_sys.evolution_fn(20, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }

    #[test]
    fn discretization_tustin_fail() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 5., 4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(2., Discretization::Tustin);
        assert!(disc_sys.is_none());
    }

    #[test]
    fn discretization_euler_backward() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::BackwardEuler).unwrap();
        //let evo = disc_sys.time_evolution(20, |_| vec![1.], &[0., 0.]);
        let evo = disc_sys.evolution_fn(50, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }

    #[test]
    fn discretization_euler_backward_fail() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 5., 4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(1., Discretization::BackwardEuler);
        assert!(disc_sys.is_none());
    }

    #[test]
    fn discretization_euler_forward() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::ForwardEuler).unwrap();
        let evo = disc_sys.evolution_fn(20, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }
}
