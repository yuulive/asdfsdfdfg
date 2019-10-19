//! # Discrete linear system
//!
//! This module contains the methods to handle discrete systems and
//! discretization of continuous systems.

use crate::linear_system::Ss;

use std::ops::{AddAssign, MulAssign, SubAssign};

use nalgebra::{ComplexField, DMatrix, DVector, Scalar};
use num_traits::Float;

/// Trait for the set of methods on discrete linear systems.
pub trait Discrete<T: Scalar> {
    //pub trait Discrete<T: Float + Mul<DMatrix<T>, Output = DMatrix<T>> + Scalar> {
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

impl<T: ComplexField + Float + Scalar> Discrete<T> for Ss<T> {
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
    let _05 = T::from_f32(0.5).unwrap(); // Safe cast to f64 of f32.
    if let Some(a) = (&identity - &sys.a * (_05 * st)).try_inverse() {
        Some(Ss {
            a: (&identity + &sys.a * (_05 * st)) * &a,
            b: &a * &sys.b * Float::sqrt(st),
            c: &sys.c * &a * Float::sqrt(st),
            d: &sys.d + &sys.c * &a * &sys.b * _05 * st,
            dim: sys.dim,
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
