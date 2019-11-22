//! Collection of commons input signals.

use num_traits::Float;

use crate::units::Seconds;

pub mod continuous {
    //! Collection of continuous signals.
    use super::*;

    /// Zero input function
    ///
    /// # Arguments
    ///
    /// * `size` - Output size
    pub fn zero<T: Float>(size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |_| vec![T::zero(); size]
    }

    /// Step function
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `size` - Output size
    pub fn step<T: Float>(k: T, size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |_| vec![k; size]
    }
}

pub mod discrete {
    //! Collection of discrete signals.
    use super::*;

    /// Zero input function
    ///
    /// # Arguments
    ///
    /// * `size` - Output size
    pub fn zero<T: Float>(size: usize) -> impl Fn(usize) -> Vec<T> {
        move |_| vec![T::zero(); size]
    }

    /// Step function
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `size` - Output size
    pub fn step<T: Float>(k: T, size: usize) -> impl Fn(usize) -> Vec<T> {
        move |_| vec![k; size]
    }

    /// Impulse function at given time
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `time` - Impulse time
    /// * `size` - Output size
    pub fn impulse<T: Float>(k: T, time: usize, size: usize) -> impl Fn(usize) -> Vec<T> {
        move |t| {
            if t == time {
                vec![k; size]
            } else {
                vec![T::zero(); size]
            }
        }
    }
}
