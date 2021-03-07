use num_traits::Zero;

use std::{
    fmt::Debug,
    ops::{Mul, Sub},
};

/// Trait representing a 2-dimensional point.
pub(super) trait Point2D {
    /// Type of abscissa and ordinate.
    type Output;
    /// Abscissa.
    fn x(&self) -> Self::Output;
    /// Ordinate.
    fn y(&self) -> Self::Output;
}

/// Difine the type of turn.
#[derive(Debug, PartialEq)]
enum Turn {
    /// Strictly left.
    Left,
    /// Strictly straight (forward or backward).
    Straight,
    /// Strictly right.
    Right,
}

/// Calculate the upper convex hull of the given set of points.
///
/// # Arguments
///
/// * `set` - set of points.
///
/// # Reference
///
/// T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein,
/// Introduction to Algorithms, 3rd edition, McGraw-Hill Education, 2009,
/// A. M. Andrew, "Another Efficient Algorithm for Convex Hulls in Two Dimensions",
/// Info. Proc. Letters 9, 216-219 (1979)
///
/// # Algorithm
///
/// Monotone chain Andrew's algorithm. The algorithm is a variant of Graham scan
/// which sorts the points lexicographically by their coordinates.
/// <https://en.wikipedia.org/wiki/Convex_hull_algorithms>
pub(super) fn convex_hull_top<I, P>(set: I) -> Vec<P>
where
    I: IntoIterator<Item = P>,
    P: Clone + Point2D,
    P::Output: Mul<Output = P::Output> + PartialOrd + Sub<Output = P::Output> + Zero,
{
    let mut iter = set.into_iter();
    let mut stack = Vec::<P>::with_capacity(2);
    if let Some(first) = iter.next() {
        stack.push(first);
    }
    if let Some(second) = iter.next() {
        stack.push(second);
    }

    // iter will continue from the 3rd element if any.
    for p in iter {
        loop {
            let length = stack.len();
            // There shall be at least 2 elements in the stack.
            if length < 2 {
                break;
            }
            let next_to_top = stack.get(length - 2).unwrap();
            let top = stack.last().unwrap();

            let turn = turn(next_to_top, top, &p);
            // Remove the top of the stack if it is not a strict turn to the right.
            match turn {
                Turn::Right => break,
                _ => stack.pop(),
            };
        }
        stack.push(p);
    }

    // stack is already sorted by k.
    stack
}

/// Define if two vectors turn right, left or are aligned.
/// First vector (p1 - p0).
/// Second vector (p2 - p0).
///
/// # Reference
///
/// T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein,
/// Introduction to Algorithms, 3rd edition, McGraw-Hill Education, 2009,
/// paragraph 33.1
fn turn<P>(p0: &P, p1: &P, p2: &P) -> Turn
where
    P: Point2D,
    P::Output: Mul<Output = P::Output> + PartialOrd + Sub<Output = P::Output> + Zero,
{
    let cp = cross_product(p0, p1, p2);
    if cp < P::Output::zero() {
        Turn::Right
    } else if cp > P::Output::zero() {
        Turn::Left
    } else {
        Turn::Straight
    }
}

/// Compute the cross product of (p1 - p0) x (p2 - p0)
///
/// `(p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)`
///
/// # Reference
///
/// T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein,
/// Introduction to Algorithms, 3rd edition, McGraw-Hill Education, 2009,
/// paragraph 33.1
fn cross_product<P>(p0: &P, p1: &P, p2: &P) -> P::Output
where
    P: Point2D,
    P::Output: Mul<Output = P::Output> + Sub<Output = P::Output>,
{
    let first_vec_x = p1.x() - p0.x();
    let first_vec_y = p1.y() - p0.y();
    let second_vec_x = p2.x() - p0.x();
    let second_vec_y = p2.y() - p0.y();
    first_vec_x * second_vec_y - second_vec_x * first_vec_y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct P(i32, i32);

    impl Point2D for P {
        type Output = i32;
        fn x(&self) -> Self::Output {
            self.0
        }
        fn y(&self) -> Self::Output {
            self.1
        }
    }

    impl Point2D for &P {
        type Output = i32;
        fn x(&self) -> Self::Output {
            self.0
        }
        fn y(&self) -> Self::Output {
            self.1
        }
    }

    #[test]
    fn vector_cross_product() {
        let cp1 = cross_product(&P(0, 0), &P(0, 1), &P(1, 0));
        assert_eq!(-1, cp1);

        let cp2 = cross_product(&P(0, 0), &P(1, 1), &P(2, 2));
        assert_eq!(0, cp2);

        let cp3 = cross_product(&P(0, 0), &P(0, -1), &P(1, 0));
        assert_eq!(1, cp3);
    }

    #[test]
    fn vector_turn() {
        let turn1 = turn(&P(0, 0), &P(0, 1), &P(1, 0));
        assert_eq!(Turn::Right, turn1);

        let turn2 = turn(&P(0, 0), &P(1, 1), &P(2, 2));
        assert_eq!(Turn::Straight, turn2);

        let turn3 = turn(&P(0, 0), &P(0, -1), &P(1, 0));
        assert_eq!(Turn::Left, turn3);

        let turn4 = turn(&P(0, 0), &P(-3, 1), &P(3, -1));
        assert_eq!(Turn::Straight, turn4);
    }

    #[test]
    fn top_hull() {
        let set = [P(0, 0), P(1, -2), P(2, 3), P(3, 3), P(4, -5)];
        let ch = convex_hull_top(&set);
        let expected = vec![&P(0, 0), &P(2, 3), &P(3, 3), &P(4, -5)];
        assert_eq!(expected, ch);
    }

    #[test]
    fn top_hull_valley() {
        let set = [P(0, 0), P(1, -2), P(2, -3), P(3, 0)];
        let ch = convex_hull_top(&set);
        let expected = vec![&P(0, 0), &P(3, 0)];
        assert_eq!(expected, ch);
    }
}
