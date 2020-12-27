use num_complex::Complex;
use num_traits::{Float, FloatConst, NumCast, One, Zero};

use std::{
    fmt::Debug,
    ops::{Mul, Sub},
};

use {super::Poly, crate::complex};

/// Default number of iterations for the iterative root finding algorithm.
pub(super) const DEFAULT_ITERATIONS: u32 = 30;

/// Structure to hold the computational data for polynomial root finding.
#[derive(Debug)]
pub(super) struct RootsFinder<T> {
    /// Polynomial
    poly: Poly<T>,
    /// Polynomial derivative
    derivative: Poly<T>,
    /// Solution, roots of the polynomial
    solution: Vec<Complex<T>>,
    /// Maximum iterations of the algorithm
    iterations: u32,
}

impl<T: Float + FloatConst + NumCast> RootsFinder<T> {
    /// Create a `RootsFinder` structure
    ///
    /// # Arguments
    ///
    /// * `poly` - polynomial whose roots have to be found.
    pub(super) fn new(poly: Poly<T>, iterations: u32) -> Self {
        let derivative = poly.derive();

        // Set the initial root approximation.
        let initial_guess = init(&poly);

        debug_assert!(poly.degree().unwrap_or(0) == initial_guess.len());

        Self {
            poly,
            derivative,
            solution: initial_guess,
            iterations,
        }
    }

    /// Algorithm to find all the complex roots of a polynomial.
    /// Iterative method that finds roots simultaneously.
    ///
    /// O. Aberth, Iteration Methods for Finding all Zeros of a Polynomial Simultaneously,
    /// Math. Comput. 27, 122 (1973) 339–344.
    ///
    /// D. A. Bini, Numerical computation of polynomial zeros by means of Aberth’s method,
    /// Baltzer Journals, June 5, 1996
    ///
    /// D. A. Bini, L. Robol, Solving secular and polynomial equations: A multiprecision algorithm,
    /// Journal of Computational and Applied Mathematics (2013)
    ///
    /// W. S. Luk, Finding roots of real polynomial simultaneously by means of Bairstow's method,
    /// BIT 35 (1995), 001-003
    pub(super) fn roots_finder(mut self) -> Vec<Complex<T>>
    where
        T: Float,
    {
        let n_roots = self.solution.len();
        let mut done = vec![false; n_roots];

        for _k in 0..self.iterations {
            if done.iter().all(|&d| d) {
                break;
            }

            for (i, d) in done.iter_mut().enumerate() {
                let solution_i = self.solution[i];
                let derivative = self.derivative.eval(&solution_i);

                let a_xki: Complex<T> = self
                    .solution
                    .iter()
                    .enumerate()
                    .filter_map(|(j, s)| {
                        // (index j, j_th solution)
                        if j == i {
                            None
                        } else {
                            let den = solution_i - s;
                            Some(den.inv())
                        }
                    })
                    .sum();

                let fraction = if derivative.is_zero() {
                    -complex::compinv(a_xki)
                } else {
                    let n_xki = complex::compdiv(self.poly.eval(&solution_i), derivative);
                    complex::compdiv(n_xki, Complex::<T>::one() - n_xki * a_xki)
                };
                // Overriding the root before updating the other decrease the time
                // the algorithm converges.
                let new = solution_i - fraction;
                *d = if solution_i == new {
                    true
                } else {
                    self.solution[i] = new;
                    false
                };
            }
        }
        self.solution
    }
}

/// Trait representing a 2-dimensional point.
trait Point2D {
    /// Type of abscissa and ordinate.
    type Output;
    /// Abscissa.
    fn x(&self) -> Self::Output;
    /// Ordinate.
    fn y(&self) -> Self::Output;
}

/// Internal struct to hold the point to calculate the convex hull
#[derive(Clone, Debug)]
struct CoeffPoint<T: Clone>(usize, T, T);

impl<T: Clone> Point2D for CoeffPoint<T> {
    type Output = T;
    fn x(&self) -> Self::Output {
        self.1.clone()
    }
    fn y(&self) -> Self::Output {
        self.2.clone()
    }
}

/// Generate the initial approximation of the polynomial roots.
///
/// Theorems 12 and 13 of D. A. Bini, L. Robol, Solving secular and polynomial
/// equations: A multiprecision algorithm, Journal of Computational and Applied Mathematics (2013)
///
/// # Arguments
///
/// * `poly` - polynomial whose roots have to be found.
///
/// # Panics
///
/// Panics if the conversion from usize to T (float) fails.
fn init<T>(poly: &Poly<T>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    // set = Iterator<Item = (k as usize, k as Float, ln(c_k) as Float)>
    let set = poly
        .coeffs
        .iter()
        .enumerate()
        .map(|(k, c)| CoeffPoint(k, T::from(k).unwrap(), c.abs().ln()));

    // Convex hull
    // ch = Vec<(k as usize, k as Float)>
    let ch: Vec<_> = convex_hull_top(set)
        .iter()
        .map(|&CoeffPoint(a, b, _)| (a, b))
        .collect();

    // Radii of the circles around which the inital roots are placed.
    // The number of roots per circle is equal to the difference between the
    // indices of consecutive coefficients on the convex hull.
    // r = Iterator<Item = (k_(i+1) - k_i as usize, r as Float)>
    let r = ch.windows(2).map(|w| {
        // w[1] = k_(i+1), w[0] = k_i
        let tmp = (poly.coeffs[w[0].0] / poly.coeffs[w[1].0]).abs();
        (w[1].0 - w[0].0, tmp.powf((w[1].1 - w[0].1).recip()))
    });

    // Initial root values.
    // For every circle of radius 'r' put 'n_k' roots on is cicumference.
    let tau = T::TAU();
    let initial: Vec<Complex<T>> = r
        .flat_map(|(n_k, r)| {
            let n_k_f = T::from(n_k).unwrap();
            (0..n_k).map(move |i| {
                let i_f = T::from(i).unwrap();
                let theta = tau * i_f / n_k_f;
                Complex::from_polar(r, theta)
            })
        })
        .collect();

    initial
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
fn convex_hull_top<I, P>(set: I) -> Vec<P>
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

/// Calculate the complex roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
#[allow(clippy::many_single_char_names)]
pub(super) fn complex_quadratic_roots_impl<T: Float>(b: T, c: T) -> (Complex<T>, Complex<T>) {
    let two = T::one() + T::one();
    let b_ = b / two;
    let d = b_.powi(2) - c; // Discriminant
    let (root1_r, root1_i, root2_r, root2_i) = if d.is_zero() {
        (-b_, T::zero(), -b_, T::zero())
    } else if d.is_sign_negative() {
        let s = (-d).sqrt();
        (-b_, -s, -b_, s)
    } else {
        // Positive discriminant.
        let s = b.signum() * d.sqrt();
        let h = -(b_ + s);
        (c / h, T::zero(), h, T::zero())
    };

    (
        Complex::new(root1_r, root1_i),
        Complex::new(root2_r, root2_i),
    )
}

/// Calculate the real roots of the quadratic equation x^2 + b*x + c = 0.
///
/// # Arguments
///
/// * `b` - first degree coefficient
/// * `c` - zero degree coefficient
#[allow(clippy::many_single_char_names)]
pub(super) fn real_quadratic_roots_impl<T: Float>(b: T, c: T) -> Option<(T, T)> {
    let two = T::one() + T::one();
    let b_ = b / two;
    let d = b_.powi(2) - c; // Discriminant
    if d.is_zero() {
        Some((-b_, -b_))
    } else if d.is_sign_negative() {
        return None;
    } else {
        // Positive discriminant.
        let s = b.signum() * d.sqrt();
        let h = -(b_ + s);
        Some((c / h, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Point(f32, f32);

    impl Point2D for Point {
        type Output = f32;
        fn x(&self) -> Self::Output {
            self.0
        }
        fn y(&self) -> Self::Output {
            self.1
        }
    }

    #[test]
    fn point_implementation() {
        let t = turn(&Point(1., 1.), &Point(2., 9.), &Point(12., -4.));
        assert_eq!(Turn::Right, t);
    }

    #[test]
    fn vector_cross_product() {
        let cp1 = cross_product(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 0, 1),
            &CoeffPoint(0, 1, 0),
        );
        assert_eq!(-1, cp1);

        let cp2 = cross_product(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 1, 1),
            &CoeffPoint(0, 2, 2),
        );
        assert_eq!(0, cp2);

        let cp3 = cross_product(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 0, -1),
            &CoeffPoint(0, 1, 0),
        );
        assert_eq!(1, cp3);
    }

    #[test]
    fn vector_turn() {
        let turn1 = turn(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 0, 1),
            &CoeffPoint(0, 1, 0),
        );
        assert_eq!(Turn::Right, turn1);

        let turn2 = turn(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 1, 1),
            &CoeffPoint(0, 2, 2),
        );
        assert_eq!(Turn::Straight, turn2);

        let turn3 = turn(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, 0, -1),
            &CoeffPoint(0, 1, 0),
        );
        assert_eq!(Turn::Left, turn3);

        let turn4 = turn(
            &CoeffPoint(0, 0, 0),
            &CoeffPoint(0, -3, 1),
            &CoeffPoint(0, 3, -1),
        );
        assert_eq!(Turn::Straight, turn4);
    }

    #[test]
    fn iterative_roots_finder() {
        let roots = &[10.0_f32, 10. / 323.4, 1., -2., 3.];
        let poly = Poly::new_from_roots(roots);
        let rf = RootsFinder::new(poly, DEFAULT_ITERATIONS);
        let actual = rf.roots_finder();
        assert_eq!(roots.len(), actual.len());
    }

    #[test]
    fn roots_finder_debug_string() {
        let poly = Poly::new_from_coeffs(&[1., 2.]);
        let rf = RootsFinder::new(poly, DEFAULT_ITERATIONS);
        let debug_str = format!("{:?}", &rf);
        assert!(
            !debug_str.is_empty(),
            "RootsFinder<T> structure must be debuggable if T: Debug."
        );
    }
}
