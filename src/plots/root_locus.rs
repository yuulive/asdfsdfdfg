//! # Root locus plot
//!
//! Trajectories of the poles when the system is put in feedback with a pure
//! constant controller

use nalgebra::RealField;
use num_complex::Complex;
use num_traits::{Float, MulAdd};

use crate::transfer_function::continuous::Tf;

/// Struct for root locus plot
#[derive(Clone, Debug)]
pub struct RootLocus<T: Float> {
    /// Transfer function
    tf: Tf<T>,
    /// Minimum transfer constant of the plot
    min_k: T,
    /// Maximum transfer constant of the plot
    max_k: T,
    /// Step size
    step: T,
}

impl<T: Float> RootLocus<T> {
    /// Create a `RootLocus` plot struct
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to plot
    /// * `min_k` - Minimum transfer constant of the plot
    /// * `max_k` - Maximum transfer constant of the plot
    /// * `step` - Step between each transfer constant
    ///
    /// `step` is linear.
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum transfer constant
    /// is not lower than the maximum transfer constant.
    pub(crate) fn new(tf: Tf<T>, min_k: T, max_k: T, step: T) -> Self {
        assert!(step > T::zero(), "Step value must be strictly positive.");
        assert!(
            min_k < max_k,
            "Maximum transfer constant must be greater than the minimum transfer constant."
        );

        Self {
            tf,
            min_k,
            max_k,
            step,
        }
    }
}

/// Struct for root locus plot
#[derive(Clone, Debug)]
pub struct IntoIter<T: Float> {
    /// Transfer function
    tf: Tf<T>,
    /// Minimum transfer constant
    min_k: T,
    /// Step size
    step: T,
    /// Number of intervals to compute
    intervals: T,
    /// Current index of iterator
    index: T,
}

impl<T: Float + MulAdd<Output = T> + RealField> IntoIterator for RootLocus<T> {
    type Item = Data<T>;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let intervals = num_traits::Float::floor((self.max_k - self.min_k) / self.step);
        Self::IntoIter {
            tf: self.tf,
            min_k: self.min_k,
            step: self.step,
            intervals,
            index: T::zero(),
        }
    }
}

/// Struct to hold the data for the root locus plot.
#[derive(Debug)]
pub struct Data<T> {
    /// Transfer constant
    k: T,
    /// Roots at the given transfer constant
    output: Vec<Complex<T>>,
}

impl<T: Copy> Data<T> {
    /// Get the transfer constant.
    pub fn k(&self) -> T {
        self.k
    }

    /// Get the roots at the given transfer constant.
    pub fn output(&self) -> &[Complex<T>] {
        &self.output
    }
}

impl<T: Float + MulAdd<Output = T> + RealField> Iterator for IntoIter<T> {
    type Item = Data<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            // k = step * index + min_k, is used to avoid loss of precision
            // of k += step, due to floating point addition
            let k = MulAdd::mul_add(self.step, self.index, self.min_k);
            self.index += T::one();
            Some(Self::Item {
                k,
                output: self.tf.root_locus(k),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly;

    #[test]
    #[should_panic]
    fn fail_new1() {
        let tf = Tf::new(poly!(1.), poly!(0., 1.));
        RootLocus::new(tf, 0.1, 0.2, 0.);
    }

    #[test]
    #[should_panic]
    fn fail_new2() {
        let tf = Tf::new(poly!(1.), poly!(0., 1.));
        RootLocus::new(tf, 0.9, 0.2, 0.1);
    }
}
