//! # Automatic Control Systems Library
//!
//! ## State-Space representation
//!
//! [State space representation](linear_system/struct.Ss.html)
//!
//! [Equilibrium](linear_system/struct.Equilibrium.html)
//!
//! ### System time evolution
//!
//! [Solvers](linear_system/solver/index.html)
//!
//! ## Discrete system
//!
//! [Discrete](linear_system/discrete/index.html)
//!
//! [Transfer function discretization](transfer_function/discrete_tf/index.html)
//!
//! ## Transfer function representation
//!
//! [Transfer function](transfer_function/struct.Tf.html)
//!
//! [Matrix of transfer functions](transfer_function/struct.TfMatrix.html)
//!
//! ## Plots
//!
//! [Bode plot](plots/bode/index.html)
//!
//! [Polar plot](plots/polar/index.html)
//!
//! ## Controllers
//!
//! [Pid](controller/pid/struct.Pid.html)
//!
//! ## Polynomials
//!
//! [Polynomials](polynomial/struct.Poly.html)
//!
//! [Matrix of polynomials](polynomial/struct.MatrixOfPoly.html)

#![warn(missing_docs)]

#[macro_use]
extern crate approx;

pub mod controller;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod transfer_function;
pub mod units;

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: &T) -> T;
}

/// Zip two slices with the given function
///
/// # Arguments
///
/// * `left` - first slice to zip
/// * `right` - second slice to zip
/// * `f` - function used to zip the two lists
#[allow(dead_code)]
pub(crate) fn zip_with<'a, L, R, T, F>(
    left: &'a [L],
    right: &'a [R],
    mut f: F,
) -> impl Iterator<Item = T> + 'a
where
    F: FnMut(&L, &R) -> T + 'a,
{
    left.iter().zip(right).map(move |(l, r)| f(l, r))
}

/// Zip two iterators extending the shorter one with the provided `fill` value.
///
/// # Arguments
///
/// * `left` - first iterator
/// * `right` - second iterator
/// * `fill` - default value
#[allow(dead_code)]
pub(crate) fn zip_longest<T: Copy>(left: &[T], right: &[T], fill: T) -> Vec<(T, T)> {
    let mut result = Vec::<(T, T)>::with_capacity(left.len().max(right.len()));
    let mut left_iter = left.iter();
    let mut right_iter = right.iter();
    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(&l), Some(&r)) => result.push((l, r)),
            (Some(&l), None) => result.push((l, fill)),
            (None, Some(&r)) => result.push((fill, r)),
            _ => break,
        }
    }
    result
}

/// Zip two iterators  with the given function extending the shorter one
/// with the provided `fill` value.
///
/// # Arguments
///
/// * `left` - first iterator
/// * `right` - second iterator
/// * `fill` - default value
/// * `f` - function used to zip the two lists
pub(crate) fn zip_longest_with<T, F>(left: &[T], right: &[T], fill: T, mut f: F) -> Vec<T>
where
    T: Copy,
    F: FnMut(T, T) -> T,
{
    let mut result = Vec::<T>::with_capacity(left.len().max(right.len()));
    let mut left_iter = left.iter();
    let mut right_iter = right.iter();
    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(&l), Some(&r)) => result.push(f(l, r)),
            (Some(&l), None) => result.push(f(l, fill)),
            (None, Some(&r)) => result.push(f(fill, r)),
            _ => break,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_longest_iterators() {
        let a = zip_longest(&[1, 2, 3, 4], &[6, 7], 0);
        assert_eq!(vec![(1, 6), (2, 7), (3, 0), (4, 0)], a);
    }

    #[test]
    fn zip_longest_with_iterators() {
        let a = zip_longest_with(&[1, 2, 3, 4], &[6, 7], 0, |x, y| x + y);
        assert_eq!(vec![7, 9, 3, 4], a);
    }
}
