//! # Discrete linear system
//!
//! This module contains the methods to handle discrete systems and
//! discretization of continuous systems.

use nalgebra::{ComplexField, DMatrix, DVector, Scalar};
use num_traits::Float;

use std::{
    marker::PhantomData,
    ops::{AddAssign, MulAssign, SubAssign},
};

use crate::{
    linear_system::{continuous::Ss, SsGen},
    Discrete,
};

/// State-space representation of discrete time linear system
pub type Ssd<T> = SsGen<T, Discrete>;

/// Trait for the set of methods on discrete linear systems.
pub trait DiscreteTime<T: Scalar> {
    /// Time evolution for a discrete linear system.
    ///
    /// # Arguments
    ///
    /// * `step` - simulation length
    /// * `input` - input function
    /// * `x0` - initial state
    fn time_evolution<F>(&self, steps: usize, input: F, x0: &[T]) -> DiscreteIterator<F, T>
    where
        F: Fn(usize) -> Vec<T>;

    /// Convert a linear system into a discrete system.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    /// * `method` - discretization method
    fn discretize(&self, st: T, method: Discretization) -> Option<Ss<T>>;
}

/// Discretization algorithm.
#[derive(Clone, Copy, Debug)]
pub enum Discretization {
    /// Forward Euler
    ForwardEuler,
    /// Backward Euler
    BackwardEuler,
    /// Tustin (trapezoidal rule)
    Tustin,
}

impl<T: ComplexField + Float + Scalar> DiscreteTime<T> for Ss<T> {
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
    /// use automatica::{linear_system::{discrete::{DiscreteTime, Discretization}}, Ss};
    /// let disc_sys = Ss::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
    /// let impulse = |t| if t == 0 { vec![1.] } else { vec![0.] };
    /// let evo = disc_sys.time_evolution(20, impulse, &[0., 0.]);
    /// let last = evo.last().unwrap();
    /// assert_abs_diff_eq!(0., last.state()[1], epsilon = 0.001);
    /// ```
    fn time_evolution<F>(&self, steps: usize, input: F, x0: &[T]) -> DiscreteIterator<F, T>
    where
        F: Fn(usize) -> Vec<T>,
    {
        let state = DVector::from_column_slice(x0);
        let next_state = DVector::from_column_slice(x0);
        DiscreteIterator {
            sys: &self,
            time: 0,
            steps,
            input,
            state,
            next_state,
        }
    }

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
    /// use automatica::{linear_system::{discrete::{DiscreteTime, Discretization}}, Ss};
    /// let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
    /// let disc_sys = sys.discretize(0.1, Discretization::Tustin).unwrap();
    /// let evo = disc_sys.time_evolution(20, |t| vec![1.], &[0., 0.]);
    /// let last = evo.last().unwrap();
    /// assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    /// ```
    fn discretize(&self, st: T, method: Discretization) -> Option<Self> {
        match method {
            Discretization::ForwardEuler => forward_euler(&self, st),
            Discretization::BackwardEuler => backward_euler(&self, st),
            Discretization::Tustin => tustin(&self, st),
        }
    }
}

/// Discretization using forward Euler Method.
///
/// # Arguments
///
/// * `sys` - continuous linear system
/// * `st` - sample time
fn forward_euler<T>(sys: &Ss<T>, st: T) -> Option<Ss<T>>
where
    T: Float + MulAssign + Scalar + AddAssign,
{
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    Some(Ss {
        a: identity + &sys.a * st,
        b: &sys.b * st,
        c: sys.c.clone(),
        d: sys.d.clone(),
        dim: sys.dim,
        time: PhantomData,
    })
}

/// Discretization using backward Euler Method.
///
/// # Arguments
///
/// * `sys` - continuous linear system
/// * `st` - sample time
fn backward_euler<T>(sys: &Ss<T>, st: T) -> Option<Ss<T>>
where
    T: ComplexField + Float + Scalar + SubAssign,
{
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    if let Some(a) = (identity - &sys.a * st).try_inverse() {
        Some(Ss {
            b: &a * &sys.b * st,
            c: &sys.c * &a,
            d: &sys.d + &sys.c * &a * &sys.b * st,
            a,
            dim: sys.dim,
            time: PhantomData,
        })
    } else {
        None
    }
}

/// Discretization using Tustin Method.
///
/// # Arguments
///
/// * `sys` - continuous linear system
/// * `st` - sample time
fn tustin<T>(sys: &Ss<T>, st: T) -> Option<Ss<T>>
where
    T: ComplexField + Float + Scalar,
{
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    // Casting is safe for both f32 and f64, representation is exact.
    let n_05 = T::from(0.5_f32).unwrap();
    if let Some(k) = (&identity - &sys.a * (n_05 * st)).try_inverse() {
        let b = &k * &sys.b * st;
        Some(Ss {
            a: &k * (&identity + &sys.a * (n_05 * st)),
            c: &sys.c * &k,
            d: &sys.d + &sys.c * &b * n_05,
            b,
            dim: sys.dim,
            time: PhantomData,
        })
    } else {
        None
    }
}

/// Struct to hold the iterator for the evolution of the discrete linear system.
#[derive(Debug)]
pub struct DiscreteIterator<'a, F, T>
where
    F: Fn(usize) -> Vec<T>,
    T: Scalar,
{
    sys: &'a Ss<T>,
    time: usize,
    steps: usize,
    input: F,
    state: DVector<T>,
    next_state: DVector<T>,
}

/// Struct to hold the result of the discrete linear system evolution.
#[derive(Debug)]
pub struct TimeEvolution<T> {
    time: usize,
    state: Vec<T>,
    output: Vec<T>,
}

impl<'a, F, T> Iterator for DiscreteIterator<'a, F, T>
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

impl<T> TimeEvolution<T> {
    /// Get the time of the current step
    pub fn time(&self) -> usize {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<T> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial::Poly;
    use std::convert::TryFrom;

    #[test]
    fn convert_to_ss_discrete() {
        use crate::transfer_function::discrete::Tfz;
        let tf = Tfz::new(
            Poly::new_from_coeffs(&[1., 0., 1.]),
            Poly::new_from_coeffs(&[3., 4., 1.]),
        );

        let ss = Ssd::try_from(tf).unwrap();

        assert_eq!(DMatrix::from_row_slice(2, 2, &[0., -3., 1., -4.]), *ss.a());
        assert_eq!(DMatrix::from_row_slice(2, 1, &[-2., -4.]), *ss.b());
        assert_eq!(DMatrix::from_row_slice(1, 2, &[0., 1.]), *ss.c());
        assert_eq!(DMatrix::from_row_slice(1, 1, &[1.]), *ss.d());
    }

    #[test]
    fn time_evolution() {
        let disc_sys =
            Ss::new_from_slice(2, 1, 1, &[0.6, 0., 0., 0.4], &[1., 5.], &[1., 3.], &[0.]);
        let impulse = |t| if t == 0 { vec![1.] } else { vec![0.] };
        let evo = disc_sys.time_evolution(20, impulse, &[0., 0.]);
        let last = evo.last().unwrap();
        assert_eq!(20, last.time());
        assert_abs_diff_eq!(0., last.state()[1], epsilon = 0.001);
        assert_abs_diff_eq!(0., last.output()[0], epsilon = 0.001);
    }

    #[test]
    fn discretization_tustin() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::Tustin).unwrap();
        let evo = disc_sys.time_evolution(20, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }

    #[test]
    fn discretization_euler_backward() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::BackwardEuler).unwrap();
        //let evo = disc_sys.time_evolution(20, |_| vec![1.], &[0., 0.]);
        let evo = disc_sys.time_evolution(50, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }

    #[test]
    fn discretization_euler_forward() {
        let sys = Ss::new_from_slice(2, 1, 1, &[-3., 0., -4., -4.], &[0., 1.], &[1., 1.], &[0.]);
        let disc_sys = sys.discretize(0.1, Discretization::ForwardEuler).unwrap();
        let evo = disc_sys.time_evolution(20, |_| vec![1.], &[0., 0.]);
        let last = evo.last().unwrap();
        assert_relative_eq!(0.25, last.state()[1], max_relative = 0.01);
    }
}
