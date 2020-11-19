//! Collection of variuous utility functions.

use num_complex::Complex;
use num_traits::Float;

use std::fmt::Debug;

/// Trait to tag Continuous or Discrete types
pub trait Time: Debug {}

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

/// Calculate the natural pulse of a complex number, it corresponds to its modulus.
///
/// # Arguments
///
/// * `c` - Complex number
///
/// # Example
/// ```
/// use num_complex::Complex;
/// use automatica::utils::pulse;
/// let i = Complex::new(0., 1.);
/// assert_eq!(1., pulse(i));
/// ```
pub fn pulse<T: Float>(c: Complex<T>) -> T {
    c.norm()
}

/// Calculate the damp of a complex number, it corresponds to the cosine of the
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
/// use automatica::utils::damp;
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

/// Zip two iterators with the given function
///
/// # Arguments
///
/// * `left` - first iterator to zip
/// * `right` - second iterator to zip
/// * `f` - function used to zip the two iterators
pub(crate) fn zip_with<L, R, T, F>(left: L, right: R, mut f: F) -> impl Iterator<Item = T>
where
    L: IntoIterator,
    R: IntoIterator,
    F: FnMut(L::Item, R::Item) -> T,
{
    left.into_iter().zip(right).map(move |(l, r)| f(l, r))
}

#[derive(Clone, Debug)]
pub(crate) struct ZipLongest<I, J>
where
    I: Iterator,
    J: Iterator,
{
    a: I,
    b: J,
    fill: I::Item,
}

/// Zip two iterators extending the shorter one with the provided `fill` value.
///
/// # Arguments
///
/// * `a` - first iterator
/// * `b` - second iterator
/// * `fill` - default value
pub(crate) fn zip_longest<I, J>(a: I, b: J, fill: I::Item) -> ZipLongest<I::IntoIter, J::IntoIter>
where
    I::Item: Clone,
    I: IntoIterator,
    J: IntoIterator<Item = I::Item>,
{
    ZipLongest {
        a: a.into_iter(),
        b: b.into_iter(),
        fill,
    }
}

impl<I, J> Iterator for ZipLongest<I, J>
where
    I::Item: Clone,
    I: Iterator,
    J: Iterator<Item = I::Item>,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.next(), self.b.next()) {
            (Some(l), Some(r)) => Some((l, r)),
            (Some(l), None) => Some((l, self.fill.clone())),
            (None, Some(r)) => Some((self.fill.clone(), r)),
            _ => None,
        }
    }
}

/// Zip two iterators  with the given function extending the shorter one
/// with the provided `fill` value.
///
/// # Arguments
///
/// * `left` - first iterator
/// * `right` - second iterator
/// * `fill` - default value
/// * `f` - function used to zip the two iterators
pub(crate) fn zip_longest_with<L, R, T, F>(
    left: L,
    right: R,
    fill: L::Item,
    mut f: F,
) -> impl Iterator<Item = T>
where
    L::Item: Clone,
    L: IntoIterator,
    R: IntoIterator<Item = L::Item>,
    F: FnMut(L::Item, L::Item) -> T,
{
    zip_longest(left, right, fill).map(move |(l, r)| f(l, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn pulse_damp() {
        let c = Complex::from_str("4+3i").unwrap();
        assert_relative_eq!(5., pulse(c));
        assert_relative_eq!(-0.8, damp(c));

        let i = Complex::from_str("i").unwrap();
        assert_relative_eq!(1., pulse(i));
        assert_relative_eq!(0., damp(i));

        let zero = Complex::from_str("0").unwrap();
        assert_relative_eq!(0., pulse(zero));
        assert_relative_eq!(-1., damp(zero));
    }

    #[test]
    fn zip_longest_left() {
        let a = zip_longest(1..=4, 6..=7, 0);
        assert_eq!(vec![(1, 6), (2, 7), (3, 0), (4, 0)], a.collect::<Vec<_>>());
    }

    #[test]
    fn zip_longest_right() {
        let a = zip_longest(['a', 'b'].iter(), ['a', 'b', 'c', 'd'].iter(), &'z');
        assert_eq!(
            vec![(&'a', &'a'), (&'b', &'b'), (&'z', &'c'), (&'z', &'d')],
            a.collect::<Vec<_>>()
        );
    }

    #[test]
    fn zip_longest_with_left() {
        let a = zip_longest_with(&[1, 2, 3, 4], &[6, 7], &0, |x, y| x + y);
        assert_eq!(vec![7, 9, 3, 4], a.collect::<Vec<_>>());
    }

    #[test]
    fn zip_longest_with_right() {
        let a = zip_longest_with(
            &[true, false],
            &[false, true, true, false],
            &true,
            |&x, &y| x && y,
        );
        assert_eq!(vec![false, false, true, false], a.collect::<Vec<_>>());
    }

    #[test]
    fn zip_longest_two_iters_copy() {
        // test zip_longest with iterator of different type
        let a = (1..6).map(|x| x * 2);
        let b = (1..7).filter(|x| x % 2 == 1);
        let res = zip_longest(a, b, 0);
        assert_eq!(
            vec![(2, 1), (4, 3), (6, 5), (8, 0), (10, 0)],
            res.collect::<Vec<_>>()
        );
    }

    #[allow(clippy::map_identity)]
    #[test]
    fn zip_longest_two_iters_non_copy() {
        // test zip_longest with iterator of different type non Copy
        let a = vec![vec![0], vec![1], vec![2]];
        let a = a.iter().map(|x| x);
        let b = vec![vec![5], vec![4]];
        let b = b.iter().filter(|_| true);
        let c = vec![10];
        let res = zip_longest(a, b, &c).map(|(x, y)| (x[0], y[0]));
        assert_eq!(vec![(0, 5), (1, 4), (2, 10)], res.collect::<Vec<_>>());
    }

    #[test]
    fn zip_longest_struct_left() {
        let mut a = zip_longest(&[1, 2, 3], &[1, 2], &0);
        assert_eq!(Some((&1, &1)), a.next());
        assert_eq!(Some((&2, &2)), a.next());
        assert_eq!(Some((&3, &0)), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_struct_collect() {
        let v1 = vec![1, 2, 3];
        let v2 = vec![1, 2];
        let a = zip_longest(&v1, &v2, &0);
        assert_eq!(vec![(&1, &1), (&2, &2), (&3, &0)], a.collect::<Vec<_>>());
    }

    #[test]
    fn zip_longest_struct_right() {
        let mut a = zip_longest(&[true, false], &[false, true, false], &true);
        assert_eq!(Some((&true, &false)), a.next());
        assert_eq!(Some((&false, &true)), a.next());
        assert_eq!(Some((&true, &false)), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_with_new_left() {
        let mut a = zip_longest_with(&[1, 2, 3], &[1, 2], &0, |x, y| x * y);
        assert_eq!(Some(1), a.next());
        assert_eq!(Some(4), a.next());
        assert_eq!(Some(0), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_with_new_right() {
        let mut a = zip_longest_with(&[true, false], &[false, true, false], &true, |&x, &y| {
            x || y
        });
        assert_eq!(Some(true), a.next());
        assert_eq!(Some(true), a.next());
        assert_eq!(Some(true), a.next());
        assert_eq!(None, a.next());
    }

    #[test]
    fn zip_longest_with_collect() {
        let v1 = vec![1, 2, 3];
        let v2 = vec![1, 2];
        let a = zip_longest_with(&v1, &v2, &0, |&x, &y| x * y);
        assert_eq!(vec![1, 4, 0], a.collect::<Vec<_>>());
    }
}
