//! Discrete linear system

use crate::linear_system::Ss;

use nalgebra::{DMatrix, DVector};

/// Trait for the set of methods on descrete linear systems.
pub trait Discrete {
    /// Time evolution for a descrete linear system.
    ///
    /// # Arguments
    ///
    /// * `step` - simulation length
    /// * `input` - input function
    /// * `x0` - initial state
    fn time_evolution<F>(
        &self,
        steps: usize,
        input: F,
        x0: &[f64],
    ) -> DiscreteIterator<F>
    where
        F: Fn(usize) -> Vec<f64>;

    /// Convert a linear system into a discrete system.
    ///
    /// # Arguments
    ///
    /// * `st` - sample time
    fn discretize(&self, st: f64, method: Discretization) -> Ss;
}

/// Discretization algorithm.
pub enum Discretization {
    /// Forward Euler
    ForwardEuler,
    /// Backward Euler
    BackwardEuler,
    /// Tustin (trapezioidal rule)
    Tustin,
}

impl Discrete for Ss {
    fn time_evolution<F>(
        &self,
        steps: usize,
        input: F,
        x0: &[f64],
    ) -> DiscreteIterator<F>
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

    fn discretize(&self, st: f64, method: Discretization) -> Ss {
        match method {
            Discretization::ForwardEuler => forward_euler(&self, st),
            Discretization::BackwardEuler => backward_euler(&self, st),
            Discretization::Tustin => tustin(&self, st),
        }
    }
}

fn forward_euler(sys: &Ss, st: f64) -> Ss {
    let states = sys.dim.0;
    let identity = DMatrix::identity(states, states);
    Ss {
        a: identity + st * sys.a.clone(),
        b: st * sys.b.clone(),
        c: sys.c.clone(),
        d: sys.d.clone(),
        dim: sys.dim,
    }
}

fn backward_euler(sys: &Ss, st: f64) -> Ss {
    let states = sys.dim.0;
    let identity = DMatrix::identity(states, states);
    let a = (identity - st * &sys.a)
        .try_inverse()
        .expect("Unable to discretize the system using backward euler method");
    Ss {
        b: st * &a * &sys.b,
        c: &sys.c * &a,
        d: &sys.d + &sys.c * &a * &sys.b * st,
        a,
        dim: sys.dim,
    }
}

fn tustin(sys: &Ss, st: f64) -> Ss {
    let states = sys.dim.0;
    let identity = DMatrix::identity(states, states);
    let a = (&identity - 0.5 * st * &sys.a)
        .try_inverse()
        .expect("Unable to discretize the system using tustin method");
    Ss {
        a: (&identity + 0.5 * st * &sys.a) * &a,
        b: &a * &sys.b * st.sqrt(),
        c: st.sqrt() * &sys.c * &a,
        d: &sys.d + &sys.c * &a * &sys.b * 0.5 * st,
        dim: sys.dim,
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
pub struct DiscreteEvolution {
    time: usize,
    state: Vec<f64>,
    output: Vec<f64>,
}

impl<'a, F> Iterator for DiscreteIterator<'a, F>
where
    F: Fn(usize) -> Vec<f64>,
{
    type Item = DiscreteEvolution;

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
            Some(DiscreteEvolution {
                time: current_time,
                state: self.state.as_slice().to_vec(),
                output: output.as_slice().to_vec(),
            })
        }
    }
}
