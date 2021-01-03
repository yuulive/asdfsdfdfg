use nalgebra::{DMatrix, RealField};
use num_complex::Complex;
use num_traits::{Float, FloatConst, Num, NumCast, One, Zero};

use std::fmt::Debug;

use {
    super::convex_hull::{self, Point2D},
    super::Poly,
    crate::complex,
};

/// Default number of iterations for the iterative root finding algorithm.
const DEFAULT_ITERATIONS: u32 = 30;

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
    let ch: Vec<_> = convex_hull::convex_hull_top(set)
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

impl<T: Float + RealField> Poly<T> {
    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    /// of the monic polynomial with opposite sign.
    fn companion(&self) -> Option<DMatrix<T>> {
        match self.degree() {
            Some(degree) if degree > 0 => {
                let hi_coeff = self.coeffs[degree];
                let comp = DMatrix::from_fn(degree, degree, |i, j| {
                    if j == degree - 1 {
                        -self.coeffs[i] / hi_coeff // monic polynomial
                    } else if i == j + 1 {
                        T::one()
                    } else {
                        T::zero()
                    }
                });
                debug_assert!(comp.is_square());
                Some(comp)
            }
            _ => None,
        }
    }

    /// Calculate the real roots of the polynomial
    /// using companion matrix eigenvalues decomposition.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let roots = &[1., -1., 0.];
    /// let p = Poly::new_from_roots(roots);
    /// assert_eq!(roots, p.real_roots().unwrap().as_slice());
    /// ```
    #[must_use]
    pub fn real_roots(&self) -> Option<Vec<T>> {
        let (zeros, cropped) = self.find_zero_roots();
        let roots = match cropped.degree() {
            Some(0) | None => None,
            Some(1) => cropped.real_deg1_root(),
            Some(2) => cropped.real_deg2_roots(),
            _ => {
                // Build the companion matrix.
                let comp = cropped.companion()?;
                comp.eigenvalues().map(|e| e.as_slice().to_vec())
            }
        };
        roots.map(|r| extend_roots(r, zeros))
    }

    /// Calculate the complex roots of the polynomial
    /// using companion matrix eigenvalues decomposition.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let i = num_complex::Complex::i();
    /// assert_eq!(vec![-i, i], p.complex_roots());
    /// ```
    #[must_use]
    pub fn complex_roots(&self) -> Vec<Complex<T>> {
        let (zeros, cropped) = self.find_zero_roots();
        let roots = match cropped.degree() {
            Some(0) | None => Vec::new(),
            Some(1) => cropped.complex_deg1_root(),
            Some(2) => cropped.complex_deg2_roots(),
            _ => {
                let comp = match cropped.companion() {
                    Some(comp) => comp,
                    None => return Vec::new(),
                };
                comp.complex_eigenvalues().as_slice().to_vec()
            }
        };
        extend_roots(roots, zeros)
    }
}

impl<T: Float + FloatConst> Poly<T> {
    /// Calculate the complex roots of the polynomial
    /// using Aberth-Ehrlich method.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let i = num_complex::Complex::i();
    /// assert_eq!(vec![-i, i], p.iterative_roots());
    /// ```
    #[must_use]
    pub fn iterative_roots(&self) -> Vec<Complex<T>> {
        self.iterative_roots_with_max(DEFAULT_ITERATIONS)
    }

    /// Calculate the complex roots of the polynomial using companion
    /// Aberth-Ehrlich method, with the given iteration limit.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - maximum number of iterations for the algorithm
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let i = num_complex::Complex::i();
    /// assert_eq!(vec![-i, i], p.iterative_roots_with_max(10));
    /// ```
    #[must_use]
    pub fn iterative_roots_with_max(&self, max_iter: u32) -> Vec<Complex<T>> {
        let (zeros, cropped) = self.find_zero_roots();
        let roots = match cropped.degree() {
            Some(0) | None => Vec::new(),
            Some(1) => cropped.complex_deg1_root(),
            Some(2) => cropped.complex_deg2_roots(),
            _ => {
                let rf = RootsFinder::new(cropped, max_iter);
                rf.roots_finder()
            }
        };
        extend_roots(roots, zeros)
    }
}

/// Extend a vector of roots of type `T` with `zeros` `Zero` elements.
///
/// # Arguments
///
/// * `roots` - Vector of roots
/// * `zeros` - Number of zeros to add
fn extend_roots<T: Clone + Zero>(mut roots: Vec<T>, zeros: usize) -> Vec<T> {
    roots.extend(std::iter::repeat(T::zero()).take(zeros));
    roots
}

impl<T: Clone + Num + Zero> Poly<T> {
    /// Remove the (multiple) zero roots from a polynomial. It returns the number
    /// of roots in zero and the polynomial without them.
    fn find_zero_roots(&self) -> (usize, Self) {
        if self.is_zero() {
            return (0, Poly::zero());
        }
        let zeros = self.zero_roots_count();
        let p = Self {
            coeffs: self.coeffs().split_off(zeros),
        };
        (zeros, p)
    }

    /// Remove the (multiple) zero roots from a polynomial in place.
    /// It returns the number of roots in zero.
    #[allow(dead_code)]
    fn find_zero_roots_mut(&mut self) -> usize {
        if self.is_zero() {
            return 0;
        }
        let zeros = self.zero_roots_count();
        self.coeffs.drain(..zeros);
        zeros
    }

    /// Count the first zero elements of the vector of coefficients.
    ///
    /// # Arguments
    ///
    /// * `vec` - slice of coefficients
    fn zero_roots_count(&self) -> usize {
        self.coeffs.iter().take_while(|c| c.is_zero()).count()
    }
}

impl<T: Float> Poly<T> {
    /// Calculate the complex roots of a polynomial of degree 1.
    pub(super) fn complex_deg1_root(&self) -> Vec<Complex<T>> {
        vec![From::from(-self[0] / self[1])]
    }

    /// Calculate the complex roots of a polynomial of degree 2.
    pub(super) fn complex_deg2_roots(&self) -> Vec<Complex<T>> {
        let b = self[1] / self[2];
        let c = self[0] / self[2];
        let (r1, r2) = complex_quadratic_roots_impl(b, c);
        vec![r1, r2]
    }

    /// Calculate the real roots of a polynomial of degree 1.
    pub(super) fn real_deg1_root(&self) -> Option<Vec<T>> {
        Some(vec![-self[0] / self[1]])
    }

    /// Calculate the real roots of a polynomial of degree 2.
    pub(super) fn real_deg2_roots(&self) -> Option<Vec<T>> {
        let b = self[1] / self[2];
        let c = self[0] / self[2];
        let (r1, r2) = real_quadratic_roots_impl(b, c)?;
        Some(vec![r1, r2])
    }
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
        None
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
    use crate::poly;
    use num_complex::Complex;

    #[test]
    fn failing_companion() {
        let p = Poly::<f32>::zero();
        assert_eq!(None, p.companion());
    }

    #[test]
    fn quadratic_roots_with_real_values() {
        let root1 = -1.;
        let root2 = -2.;
        assert_eq!(Some((root1, root2)), real_quadratic_roots_impl(3., 2.));

        let root3 = 1.;
        let root4 = 2.;
        assert_eq!(Some((root3, root4)), real_quadratic_roots_impl(-3., 2.));

        assert_eq!(None, real_quadratic_roots_impl(-6., 10.));

        let root5 = 3.;
        assert_eq!(Some((root5, root5)), real_quadratic_roots_impl(-6., 9.));
    }

    #[test]
    fn real_1_root_eigen() {
        let p = poly!(10., -2.);
        let r = p.real_roots().unwrap();
        assert_eq!(r.len(), 1);
        assert_relative_eq!(5., r[0]);
    }

    #[test]
    fn real_3_roots_eigen() {
        let roots = &[-1., 0., 1.];
        let p = Poly::new_from_roots(roots);
        let mut sorted_roots = p.real_roots().unwrap();
        sorted_roots.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        for (r, rr) in roots.iter().zip(&sorted_roots) {
            assert_relative_eq!(*r, *rr);
        }
    }

    #[test]
    fn complex_1_root_eigen() {
        let p = poly!(10., -2.);
        let r = p.complex_roots();
        assert_eq!(r.len(), 1);
        assert_eq!(Complex::new(5., 0.), r[0]);
    }

    #[test]
    fn complex_3_roots_eigen() {
        let p = Poly::new_from_coeffs(&[1.0_f32, 0., 1.]) * poly!(2., 1.);
        assert_eq!(p.complex_roots().len(), 3);
    }

    #[test]
    fn complex_2_roots() {
        let root1 = Complex::<f64>::new(-1., 0.);
        let root2 = Complex::<f64>::new(-2., 0.);
        assert_eq!((root1, root2), complex_quadratic_roots_impl(3., 2.));

        let root1 = Complex::<f64>::new(1., 0.);
        let root2 = Complex::<f64>::new(2., 0.);
        assert_eq!((root1, root2), complex_quadratic_roots_impl(-3., 2.));

        let root1 = Complex::<f64>::new(-0., -1.);
        let root2 = Complex::<f64>::new(-0., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots_impl(0., 1.));

        let root1 = Complex::<f64>::new(3., -1.);
        let root2 = Complex::<f64>::new(3., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots_impl(-6., 10.));

        let root1 = Complex::<f64>::new(3., 0.);
        assert_eq!((root1, root1), complex_quadratic_roots_impl(-6., 9.));
    }

    #[test]
    fn none_roots_iterative() {
        let p: Poly<f32> = Poly::zero();
        let res = p.iterative_roots();
        assert_eq!(0, res.len());
        assert!(res.is_empty());

        let p = poly!(5.3);
        let res = p.iterative_roots();
        assert_eq!(0, res.len());
        assert!(res.is_empty());
    }

    #[test]
    fn complex_1_roots_iterative() {
        let root = -12.4;
        let p = poly!(3.0 * root, 3.0);
        let res = p.iterative_roots();
        assert_eq!(1, res.len());
        let expected: Complex<f64> = From::from(-root);
        assert_eq!(expected, res[0]);
    }

    #[test]
    fn complex_2_roots_iterative() {
        let p = poly!(6., 5., 1.);
        let res = p.iterative_roots();
        assert_eq!(2, res.len());
        let expected1: Complex<f64> = From::from(-3.);
        let expected2: Complex<f64> = From::from(-2.);
        assert_eq!(expected2, res[0]);
        assert_eq!(expected1, res[1]);
    }

    #[test]
    fn complex_3_roots_iterative() {
        let p = Poly::new_from_coeffs(&[1.0_f32, 0., 1.]) * poly!(2., 1.);
        assert_eq!(p.iterative_roots().len(), 3);
    }

    #[test]
    fn complex_3_roots_with_zeros_iterative() {
        let p = Poly::new_from_coeffs(&[0.0_f32, 0., 1.]) * poly!(2., 1.);
        let mut roots = p.iterative_roots();
        assert_eq!(roots.len(), 3);
        assert_eq!(*roots.last().unwrap(), Complex::zero());
        roots.pop();
        assert_eq!(*roots.last().unwrap(), Complex::zero());
    }

    #[test]
    fn none_roots_iterative_with_max() {
        let p: Poly<f32> = Poly::zero();
        let res = p.iterative_roots_with_max(5);
        assert_eq!(0, res.len());
        assert!(res.is_empty());

        let p = poly!(5.3);
        let res = p.iterative_roots_with_max(6);
        assert_eq!(0, res.len());
        assert!(res.is_empty());
    }

    #[test]
    fn complex_1_roots_iterative_with_max() {
        let root = -12.4;
        let p = poly!(3.0 * root, 3.0);
        let res = p.iterative_roots_with_max(5);
        assert_eq!(1, res.len());
        let expected: Complex<f64> = From::from(-root);
        assert_eq!(expected, res[0]);
    }

    #[test]
    fn complex_2_roots_iterative_with_max() {
        let p = poly!(6., 5., 1.);
        let res = p.iterative_roots_with_max(6);
        assert_eq!(2, res.len());
        let expected1: Complex<f64> = From::from(-3.);
        let expected2: Complex<f64> = From::from(-2.);
        assert_eq!(expected2, res[0]);
        assert_eq!(expected1, res[1]);
    }

    #[test]
    fn complex_3_roots_iterative_with_max() {
        let p = Poly::new_from_coeffs(&[1.0_f32, 0., 1.]) * poly!(2., 1.);
        assert_eq!(p.iterative_roots_with_max(7).len(), 3);
    }

    #[test]
    fn remove_zero_roots() {
        let p = Poly::new_from_coeffs(&[0, 0, 1, 0, 2]);
        let (z, p2) = p.find_zero_roots();
        assert_eq!(2, z);
        assert_eq!(Poly::new_from_coeffs(&[1, 0, 2]), p2);
    }

    #[test]
    fn remove_zero_roots_mut() {
        let mut p = Poly::new_from_coeffs(&[0, 0, 1, 0, 2]);
        let z = p.find_zero_roots_mut();
        assert_eq!(2, z);
        assert_eq!(Poly::new_from_coeffs(&[1, 0, 2]), p);

        assert_eq!(0, Poly::<i8>::zero().find_zero_roots_mut());
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

    #[allow(clippy::float_cmp)]
    #[test]
    fn coeffpoint_implementation() {
        let cp = &CoeffPoint(1, 2., -3.);
        assert_eq!(2., cp.x());
        assert_eq!(-3., cp.y());
    }
}
