//! Root locus plot

use nalgebra::{ComplexField, RealField, Scalar};
use num_complex::Complex;
use num_traits::{Float, MulAdd};

use crate::transfer_function::continuous::Tf;

/// Struct for root locus plot
#[derive(Debug)]
pub struct RootLocusIterator<T: Float> {
    /// Transfer function
    tf: Tf<T>,
    /// Transfer constant
    min_k: T,
    intervals: T,
    step: T,
    index: T,
}

impl<T: Float> RootLocusIterator<T> {
    /// Create a RootLocusIterator struct
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
        assert!(step > T::zero());
        assert!(min_k < max_k);

        let intervals = ((max_k - min_k) / step).floor();
        Self {
            tf,
            min_k,
            intervals,
            step,
            index: T::zero(),
        }
    }
}

/// Struct to hold the data for the root locus plot.
#[derive(Debug)]
pub struct RootLocus<T: Float> {
    /// Transfer constant
    k: T,
    /// Roots at the given transfer constant
    output: Vec<Complex<T>>,
}

impl<T: Float> RootLocus<T> {
    /// Get the transfer constant.
    pub fn k(&self) -> T {
        self.k
    }

    /// Get the roots at the given transfer constant.
    pub fn output(&self) -> &[Complex<T>] {
        &self.output
    }
}

impl<T: ComplexField + Float + MulAdd<Output = T> + RealField + Scalar> Iterator
    for RootLocusIterator<T>
{
    type Item = RootLocus<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.intervals {
            None
        } else {
            let k = MulAdd::mul_add(self.step, self.index, self.min_k);
            self.index += T::one();
            Some(Self::Item {
                k,
                output: self.tf.root_locus(k),
            })
        }
    }
}
