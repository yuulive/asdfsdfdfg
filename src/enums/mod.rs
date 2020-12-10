//! Enumerations for general use inside the library.

use std::fmt::Debug;

/// Trait to tag Continuous or Discrete types
pub trait Time: Clone + Debug {}

/// Type for continuous systems
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Continuous {}
impl Time for Continuous {}

/// Type for discrete systems
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Discrete {}
impl Time for Discrete {}

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
