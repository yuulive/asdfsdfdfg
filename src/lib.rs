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

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod controller;
pub mod linear_system;
pub mod plots;
pub mod polynomial;
pub mod signals;
pub mod transfer_function;
pub mod units;

pub use linear_system::{continuous::Ss, discrete::Ssd};
pub use polynomial::Poly;
pub use transfer_function::{continuous::Tf, discrete::Tfz, matrix::TfMatrix};

use num_complex::Complex;
use num_traits::Float;

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: &T) -> T;
}

/// Trait to tag Continuous or Discrete types
pub trait Time {}

/// Type for continuous systems
#[derive(Debug, PartialEq)]
pub enum Continuous {}
impl Time for Continuous {}

/// Type for discrete systems
#[derive(Debug, PartialEq)]
pub enum Discrete {}
impl Time for Discrete {}

/// Calcualte the natural pulse of a complex number, it corresponds to its modulus.
///
/// # Arguments
///
/// * `c` - Complex number
///
/// # Example
/// ```
/// use num_complex::Complex;
/// use automatica::pulse;
/// let i = Complex::new(0., 1.);
/// assert_eq!(1., pulse(i));
/// ```
pub fn pulse<T: Float>(c: Complex<T>) -> T {
    c.norm()
}

/// Calcualte the damp of a complex number, it corresponds to the cosine of the
/// angle between the segment joining the complex number to the origin and the
/// real negative semiaxis.
///
/// By definition the damp of 0+0i is -1.
///
/// # Arguments
///
/// * `c` - Complex number
///
/// # Example
/// ```
/// use num_complex::Complex;
/// use automatica::damp;
/// let i = Complex::new(0., 1.);
/// assert_eq!(0., damp(i));
/// ```
pub fn damp<T: Float>(c: Complex<T>) -> T {
    let w = c.norm();
    if w == T::zero() {
        // Handle the case where the pusle is zero to avoid division by zero.
        -T::one()
    } else {
        -c.re / w
    }
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

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct ZipLongest<T, I, J>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
{
    a: I,
    b: J,
    fill: T,
}

#[allow(dead_code)]
fn zip_lo<T, I, J>(a: I, b: J, fill: T) -> ZipLongest<T, I::IntoIter, J::IntoIter>
where
    I: IntoIterator<Item = T>,
    J: IntoIterator<Item = T>,
{
    ZipLongest {
        a: a.into_iter(),
        b: b.into_iter(),
        fill,
    }
}

#[allow(dead_code)]
impl<T, I, J> Iterator for ZipLongest<T, I, J>
where
    T: Copy,
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (Some(l), Some(r)) => Some((l, r)),
            (Some(l), None) => Some((l, self.fill)),
            (None, Some(r)) => Some((self.fill, r)),
            _ => None,
        }
    }
}

#[allow(dead_code)]
pub(crate) fn zip_longest_with_new<'a, U, T, F>(
    left: &'a [U],
    right: &'a [U],
    fill: &'a U,
    mut f: F,
) -> impl Iterator<Item = T> + 'a
where
    F: FnMut(U, U) -> T + 'a,
    U: Copy + 'a,
{
    zip_lo(left.iter(), right.iter(), fill).map(move |(&l, &r)| f(l, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn pulse_damp() {
        let c = Complex::from_str("4+3i").unwrap();
        assert_eq!(5., pulse(c));
        assert_eq!(-0.8, damp(c));

        let i = Complex::from_str("i").unwrap();
        assert_eq!(1., pulse(i));
        assert_eq!(0., damp(i));

        let zero = Complex::from_str("0").unwrap();
        assert_eq!(0., pulse(zero));
        assert_eq!(-1., damp(zero));
    }

    #[test]
    fn zip_longest_left() {
        let a = zip_longest(&[1, 2, 3, 4], &[6, 7], 0);
        assert_eq!(vec![(1, 6), (2, 7), (3, 0), (4, 0)], a);
    }

    #[test]
    fn zip_longest_right() {
        let a = zip_longest(&['a', 'b'], &['a', 'b', 'c', 'd'], 'z');
        assert_eq!(vec![('a', 'a'), ('b', 'b'), ('z', 'c'), ('z', 'd')], a);
    }

    #[test]
    fn zip_longest_with_left() {
        let a = zip_longest_with(&[1, 2, 3, 4], &[6, 7], 0, |x, y| x + y);
        assert_eq!(vec![7, 9, 3, 4], a);
    }

    #[test]
    fn zip_longest_with_right() {
        let a = zip_longest_with(&[true, false], &[false, true, true, false], true, |x, y| {
            x && y
        });
        assert_eq!(vec![false, false, true, false], a);
    }

    #[test]
    fn zip_longest_struct_left() {
        let mut a = zip_lo(&[1, 2, 3], &[1, 2], &0);
        assert_eq!(Some((&1, &1)), a.next());
        assert_eq!(Some((&2, &2)), a.next());
        assert_eq!(Some((&3, &0)), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_struct_right() {
        let mut a = zip_lo(&[true, false], &[false, true, false], &true);
        assert_eq!(Some((&true, &false)), a.next());
        assert_eq!(Some((&false, &true)), a.next());
        assert_eq!(Some((&true, &false)), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_with_new_left() {
        let mut a = zip_longest_with_new(&[1, 2, 3], &[1, 2], &0, |x, y| x * y);
        assert_eq!(Some(1), a.next());
        assert_eq!(Some(4), a.next());
        assert_eq!(Some(0), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_with_new_right() {
        let mut a =
            zip_longest_with_new(&[true, false], &[false, true, false], &true, |x, y| x || y);
        assert_eq!(Some(true), a.next());
        assert_eq!(Some(true), a.next());
        assert_eq!(Some(true), a.next());
        assert_eq!(None, a.next());
    }
}
