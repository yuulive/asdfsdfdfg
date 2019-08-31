//! # Discrete linear system
//!
//! This module contains the methods to handle discrete systems and
//! discretization of continuous systems.

use crate::linear_system::Ss;

use nalgebra::{DMatrix, DVector};

/// Trait for the set of methods on discrete linear systems.
pub trait Discrete {
    /// Time evolution for a discrete linear system.
    ///
    /// # Arguments
    ///
    /// * `step` - simulation length
    /// * `input` - input function
    /// * `x0` - initial state
    fn time_evolution<F>(&self, steps: usize, input: F, x0: &[f64]) -> DiscreteIterator<F>
    where
        F: Fn(usize) -> Vec<f64>;

    /// Convert a linear system into a discrete system.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    fn discretize(&self, st: f64, method: Discretization) -> Option<Ss>;
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

impl Discrete for Ss {
    fn time_evolution<F>(&self, steps: usize, input: F, x0: &[f64]) -> DiscreteIterator<F>
    where
        F: Fn(usize) -> Vec<f64>,
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

    fn discretize(&self, st: f64, method: Discretization) -> Option<Ss> {
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
fn forward_euler(sys: &Ss, st: f64) -> Option<Ss> {
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    Some(Ss {
        a: identity + st * &sys.a,
        b: st * &sys.b,
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
fn backward_euler(sys: &Ss, st: f64) -> Option<Ss> {
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    if let Some(a) = (identity - st * &sys.a).try_inverse() {
        Some(Ss {
            b: st * &a * &sys.b,
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
fn tustin(sys: &Ss, st: f64) -> Option<Ss> {
    let states = sys.dim.states;
    let identity = DMatrix::identity(states, states);
    if let Some(a) = (&identity - 0.5 * st * &sys.a).try_inverse() {
        Some(Ss {
            a: (&identity + 0.5 * st * &sys.a) * &a,
            b: &a * &sys.b * st.sqrt(),
            c: st.sqrt() * &sys.c * &a,
            d: &sys.d + &sys.c * &a * &sys.b * 0.5 * st,
            dim: sys.dim,
        })
    } else {
        None
    }
}

/// Struct to hold the iterator for the evolution of the discrete linear system.
#[derive(Debug)]
pub struct DiscreteIterator<'a, F>
where
    F: Fn(usize) -> Vec<f64>,
{
    sys: &'a Ss,
    time: usize,
    steps: usize,
    input: F,
    state: DVector<f64>,
    next_state: DVector<f64>,
}

/// Struct to hold the result of the discrete linear system evolution.
#[derive(Debug)]
pub struct TimeEvolution {
    time: usize,
    state: Vec<f64>,
    output: Vec<f64>,
}

impl<'a, F> Iterator for DiscreteIterator<'a, F>
where
    F: Fn(usize) -> Vec<f64>,
{
    type Item = TimeEvolution;

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

impl TimeEvolution {
    /// Get the time of the current step
    pub fn time(&self) -> usize {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<f64> {
        &self.output
    }
}
