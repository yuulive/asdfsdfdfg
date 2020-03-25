//! # Polynomials
//!
//! Polynomial implementation
//! * builder from coefficients or roots
//! * degree
//! * extend by adding 0 coefficients to higher order terms
//! * arithmetic operations between polynomials (addition, subtraction,
//!   multiplication, division, reminder, negation)
//! * arithmetic operations with floats (addition, subtraction,
//!   multiplication, division)
//! * transformation to monic form
//! * roots finding (real and complex) using eigenvalues of the companion matrix
//! * differentiation and integration
//! * evaluation using real or complex numbers
//! * coefficient indexing
//! * zero and unit polynomials

pub mod matrix;

use nalgebra::{ComplexField, DMatrix, RealField, Scalar, Schur};
use num_complex::Complex;
use num_traits::{Float, FloatConst, MulAdd, Num, NumCast, One, Signed, Zero};

use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Rem, Sub},
};

use crate::{polynomial::matrix::PolyMatrix, utils, Eval};

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the highest degree
///
/// `p(x) = c0 + c1*x + c2*x^2 + ...`
#[derive(Debug, PartialEq, Clone)]
pub struct Poly<T> {
    pub(crate) coeffs: Vec<T>,
}

/// Macro shortcut to crate a polynomial from its coefficients.
///
/// # Example
/// ```
/// #[macro_use] extern crate automatica;
/// let p = poly!(1., 2., 3.);
/// assert_eq!(Some(2), p.degree());
/// ```
#[macro_export]
macro_rules! poly {
    ($($c:expr),+ $(,)*) => {
        $crate::polynomial::Poly::new_from_coeffs(&[$($c,)*]);
    };
}

/// Implementation methods for Poly struct
impl<T> Poly<T> {
    /// Length of the polynomial coefficients
    pub(crate) fn len(&self) -> usize {
        self.coeffs.len()
    }
}

/// Implementation methods for Poly struct
impl<T: Copy> Poly<T> {
    /// Vector of the polynomial's coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(vec![1., 2., 3.], p.coeffs());
    /// ```
    #[must_use]
    pub fn coeffs(&self) -> Vec<T> {
        self.coeffs.clone()
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Num + Zero> Poly<T> {
    /// Degree of the polynomial
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// assert_eq!(Some(2), p.degree());
    /// ```
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        assert!(
            !self.coeffs.is_empty(),
            "Degree is not defined on empty polynomial"
        );
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Extend the polynomial coefficients with 0 to the given degree in place.
    /// It does not truncate the polynomial.
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of the new highest coefficient.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let mut p = Poly::new_from_coeffs(&[1, 2, 3]);
    /// p.extend(5);
    /// assert_eq!(vec![1, 2, 3, 0, 0, 0], p.coeffs());
    /// ```
    pub fn extend(&mut self, degree: usize) {
        match self.degree() {
            None => self.coeffs.resize(degree + 1, T::zero()),
            Some(d) if degree > d => self.coeffs.resize(degree + 1, T::zero()),
            _ => (),
        };
    }
}

impl<T: Copy + Div<Output = T>> Poly<T> {
    /// In place division with a real number
    ///
    /// # Arguments
    ///
    /// * `d` - Real number divisor
    ///
    /// # Example
    /// ```
    /// use automatica::poly;
    /// let mut p = poly!(3, 4, 5);
    /// p.div_mut(2);
    /// assert_eq!(poly!(1, 2, 2), p);
    /// ```
    pub fn div_mut(&mut self, d: T) {
        for c in &mut self.coeffs {
            *c = *c / d;
        }
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Div<Output = T> + One> Poly<T> {
    /// Return the monic polynomial and the leading coefficient.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let (p2, c) = p.monic();
    /// assert_eq!(Poly::new_from_coeffs(&[0.1, 0.2, 1.]), p2);
    /// assert_eq!(10., c);
    /// ```
    #[must_use]
    pub fn monic(&self) -> (Self, T) {
        let lc = self.leading_coeff();
        let result: Vec<_> = self.coeffs.iter().map(|&x| x / lc).collect();
        let monic_poly = Self { coeffs: result };

        (monic_poly, lc)
    }

    /// Return the monic polynomial and the leading coefficient,
    /// it mutates the polynomial in place.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let mut p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let c = p.monic_mut();
    /// assert_eq!(Poly::new_from_coeffs(&[0.1, 0.2, 1.]), p);
    /// assert_eq!(10., c);
    /// ```
    pub fn monic_mut(&mut self) -> T {
        let lc = self.leading_coeff();
        self.div_mut(lc);
        lc
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + One> Poly<T> {
    /// Return the leading coefficient of the polynomial.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 10.]);
    /// let c = p.leading_coeff();
    /// assert_eq!(10., c);
    /// ```
    #[must_use]
    pub fn leading_coeff(&self) -> T {
        *self.coeffs.last().unwrap_or(&T::one())
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + PartialEq + Zero> Poly<T> {
    /// Create a new polynomial given a slice of real coefficients.
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 2., 3.]);
    /// ```
    pub fn new_from_coeffs(coeffs: &[T]) -> Self {
        let mut p = Self {
            coeffs: coeffs.into(),
        };
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }

    /// Trim the zeros coefficients of high degree terms.
    /// It will not leave an empty `coeffs` vector: zero poly is returned.
    fn trim(&mut self) {
        // TODO try to use assert macro.
        //.rposition(|&c| relative_ne!(c, 0.0, epsilon = epsilon, max_relative = max_relative))
        if let Some(p) = self.coeffs.iter().rposition(|&c| c != T::zero()) {
            let new_length = p + 1;
            self.coeffs.truncate(new_length);
        } else {
            self.coeffs.resize(1, T::zero());
        }
    }
}

/// Implementation methods for Poly struct
impl<T: AddAssign + Copy + Num + Neg<Output = T>> Poly<T> {
    /// Create a new polynomial given a slice of real roots
    /// It trims any leading zeros in the high order coefficients.
    ///
    /// # Arguments
    ///
    /// * `roots` - slice of roots
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_roots(&[1., 2., 3.]);
    /// ```
    pub fn new_from_roots(roots: &[T]) -> Self {
        let mut p = roots.iter().fold(Self::one(), |acc, &r| {
            acc * Self {
                coeffs: vec![-r, T::one()],
            }
        });
        p.trim();
        debug_assert!(!p.coeffs.is_empty());
        p
    }
}

/// Implementation methods for Poly struct
impl<T: ComplexField + Float + RealField + Scalar> Poly<T> {
    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    /// of the monic polynomial with opposite sign.
    pub(crate) fn companion(&self) -> Option<DMatrix<T>> {
        match self.degree() {
            Some(degree) if degree > 0 => {
                let hi_coeff = self.coeffs[degree];
                Some(DMatrix::from_fn(degree, degree, |i, j| {
                    if j == degree - 1 {
                        -self.coeffs[i] / hi_coeff // monic polynomial
                    } else if i == j + 1 {
                        T::one()
                    } else {
                        T::zero()
                    }
                }))
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
    /// let roots = &[-1., 1., 0.];
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
                let schur = Schur::new(comp);
                schur.eigenvalues().map(|e| e.as_slice().to_vec())
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
                    _ => return Vec::new(),
                };
                let schur = Schur::new(comp);
                schur.complex_eigenvalues().as_slice().to_vec()
            }
        };
        extend_roots(roots, zeros)
    }
}

impl<T: Float + FloatConst + MulAdd<Output = T>> Poly<T> {
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
        let (zeros, cropped) = self.find_zero_roots();
        let roots = match cropped.degree() {
            Some(0) | None => Vec::new(),
            Some(1) => cropped.complex_deg1_root(),
            Some(2) => cropped.complex_deg2_roots(),
            _ => {
                let rf = RootsFinder::new(cropped);
                rf.roots_finder()
            }
        };
        extend_roots(roots, zeros)
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
                let rf = RootsFinder::new(cropped).with_max_iterations(max_iter);
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

impl<T: Copy + Num + Zero> Poly<T> {
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
    fn complex_deg1_root(&self) -> Vec<Complex<T>> {
        vec![From::from(-self[0] / self[1])]
    }

    /// Calculate the complex roots of a polynomial of degree 2.
    fn complex_deg2_roots(&self) -> Vec<Complex<T>> {
        let b = self[1] / self[2];
        let c = self[0] / self[2];
        let (r1, r2) = complex_quadratic_roots(b, c);
        vec![r1, r2]
    }

    /// Calculate the real roots of a polynomial of degree 1.
    fn real_deg1_root(&self) -> Option<Vec<T>> {
        Some(vec![-self[0] / self[1]])
    }

    /// Calculate the real roots of a polynomial of degree 2.
    fn real_deg2_roots(&self) -> Option<Vec<T>> {
        let b = self[1] / self[2];
        let c = self[0] / self[2];
        let (r1, r2) = real_quadratic_roots(b, c)?;
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
pub(crate) fn complex_quadratic_roots<T: Float>(b: T, c: T) -> (Complex<T>, Complex<T>) {
    let b_ = b / T::from(2.0_f32).unwrap(); // Safe cast, it's exact.
    let d = b_.powi(2) - c; // Discriminant
    let (root1_r, root1_i, root2_r, root2_i) = if d.is_zero() {
        (-b_, T::zero(), -b_, T::zero())
    } else if d.is_sign_negative() {
        // Negative discriminant.
        let s = (-d).sqrt();
        (-b_, -s, -b_, s)
    } else {
        // Positive discriminant.
        let s = d.sqrt();
        let g = if b > T::zero() { T::one() } else { -T::one() };
        let h = -(b_ + g * s);
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
pub(crate) fn real_quadratic_roots<T: Float>(b: T, c: T) -> Option<(T, T)> {
    let b_ = b / T::from(2.0_f32).unwrap(); // Safe cast, it's exact.
    let d = b_.powi(2) - c; // Discriminant
    let (r1, r2) = if d.is_zero() {
        (-b_, -b_)
    } else if d.is_sign_negative() {
        return None;
    } else {
        // Positive discriminant.
        let s = d.sqrt();
        let g = if b > T::zero() { T::one() } else { -T::one() };
        let h = -(b_ + g * s);
        (c / h, h)
    };

    Some((r1, r2))
}

/// Implementation methods for Poly struct
impl Poly<f64> {
    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn multiply(&self, rhs: &DMatrix<f64>) -> PolyMatrix<f64> {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

impl Poly<f32> {
    /// Implementation of polynomial and matrix multiplication
    pub(crate) fn multiply(&self, rhs: &DMatrix<f32>) -> PolyMatrix<f32> {
        // It's the polynomial matrix whose coefficients are the coefficients
        // of the polynomial times the matrix
        let result: Vec<_> = self.coeffs.iter().map(|&c| c * rhs).collect();
        PolyMatrix::new_from_coeffs(&result)
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Mul<Output = T> + NumCast + One> Poly<T> {
    /// Calculate the derivative of the polynomial.
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 1.]);
    /// let d = p.derive();
    /// assert_eq!(Poly::new_from_coeffs(&[0., 2.]), d);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the exponent of the term (`usize`) cannot be converted
    /// to `T`.
    #[must_use]
    pub fn derive(&self) -> Self {
        let derive_coeffs: Vec<_> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| *c * T::from(i).unwrap())
            .collect();
        Self {
            coeffs: derive_coeffs,
        }
    }
}

/// Implementation methods for Poly struct
impl<T: Copy + Div<Output = T> + NumCast> Poly<T> {
    /// Calculate the integral of the polynomial. When used with integral types
    /// it does not convert the coefficients to floats, division is between
    /// integers.
    ///
    /// # Arguments
    ///
    /// * `constant` - Integration constant
    ///
    /// # Example
    /// ```
    /// use automatica::polynomial::Poly;
    /// let p = Poly::new_from_coeffs(&[1., 0., 3.]);
    /// let d = p.integrate(5.3);
    /// assert_eq!(Poly::new_from_coeffs(&[5.3, 1., 0., 1.]), d);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the exponent of the term (`usize`) cannot be converted
    /// to `T`.
    pub fn integrate(&self, constant: T) -> Self {
        let int_coeffs: Vec<_> = std::iter::once(constant)
            .chain(
                self.coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| *c / T::from(i + 1).unwrap()),
            )
            .collect();
        Self { coeffs: int_coeffs }
    }
}

/// Evaluate the polynomial at the given real or complex number
// impl<N, T> Eval<N> for Poly<T>
// where
//     N: Copy + MulAdd<Output = N> + NumCast + Zero,
//     T: Copy + NumCast,
// {
//     /// Evaluate the polynomial using Horner's method. The evaluation is safe
//     /// if the polynomial coefficient can be casted the type `N`.
//     ///
//     /// # Arguments
//     ///
//     /// * `x` - Value at which the polynomial is evaluated.
//     ///
//     /// # Panics
//     ///
//     /// The method panics if the conversion from `T` to type `N` fails.
//     ///
//     /// # Example
//     /// ```
//     /// use automatica::{Eval, polynomial::Poly};
//     /// use num_complex::Complex;
//     /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
//     /// assert_eq!(18., p.eval(3.));
//     /// assert_eq!(Complex::new(-18., 0.), p.eval(Complex::new(0., 3.)));
//     /// ```
//     fn eval_ref(&self, x: &N) -> N {
//         self.coeffs
//             .iter()
//             .rev()
//             .fold(N::zero(), |acc, &c| acc.mul_add(*x, N::from(c).unwrap()))
//     }
// }
impl<N, T> Eval<N> for Poly<T>
where
    N: Add<T, Output = N> + Copy + Mul<Output = N> + Zero,
    T: Copy,
{
    // The current implementation relies on the ability to add type N and T.
    // When the trait MulAdd<N,T> for N=Complex<T>, mul_add may be used.

    /// Evaluate the polynomial using Horner's method.
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated.
    ///
    /// # Example
    /// ```
    /// use automatica::{Eval, polynomial::Poly};
    /// use num_complex::Complex;
    /// let p = Poly::new_from_coeffs(&[0., 0., 2.]);
    /// assert_eq!(18., p.eval(3.));
    /// assert_eq!(Complex::new(-18., 0.), p.eval(Complex::new(0., 3.)));
    /// ```
    fn eval_ref(&self, x: &N) -> N {
        self.coeffs
            .iter()
            .rev()
            .fold(N::zero(), |acc, &c| acc * *x + c)
    }
}

/// Structure to hold the computational data for polynomial root finding.
#[derive(Debug)]
struct RootsFinder<T> {
    /// Polynomial
    poly: Poly<T>,
    /// Polynomial derivative
    der: Poly<T>,
    /// Solution, roots of the polynomial
    solution: Vec<Complex<T>>,
    /// Maximum iterations of the algorithm
    iterations: u32,
}

impl<T: Float + FloatConst + MulAdd<Output = T> + NumCast> RootsFinder<T> {
    /// Create a `RootsFinder` structure
    ///
    /// # Arguments
    ///
    /// * `poly` - polynomial whose roots have to be found.
    fn new(poly: Poly<T>) -> Self {
        let der = poly.derive();

        // Set the initial root approximation.
        let initial_guess = init(&poly);

        Self {
            poly,
            der,
            solution: initial_guess,
            iterations: 30,
        }
    }

    /// Define the maximum number of iterations
    ///
    /// # Arguments
    ///
    /// * `iterations` - maximum number of iterations.
    fn with_max_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
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
    fn roots_finder(mut self) -> Vec<Complex<T>>
    where
        T: Float + MulAdd<Output = T>,
    {
        let n_roots = self.poly.degree().unwrap_or(0);
        let mut done = vec![false; n_roots];

        for _k in 0..self.iterations {
            if done.iter().all(|&d| d) {
                break;
            }

            for i in 0..n_roots {
                let solution_i = *self.solution.get(i).unwrap();
                let n_xki = self.poly.eval(solution_i) / self.der.eval(solution_i);
                let a_xki: Complex<T> = (0..n_roots)
                    .filter_map(|j| {
                        if j == i {
                            None
                        } else {
                            let den = solution_i - self.solution.get(j).unwrap();
                            Some(den.inv() * T::one())
                        }
                    })
                    .sum();

                // Overriding the root before updating the other decrease the time
                // the algorithm converges.
                let new = solution_i - n_xki / (Complex::<T>::one() - n_xki * a_xki);
                *done.get_mut(i).unwrap() = if solution_i == new {
                    true
                } else {
                    *self.solution.get_mut(i).unwrap() = new;
                    false
                };
            }
        }
        self.solution
    }
}

/// Simple initialization of roots
///
/// # Arguments
///
/// * `poly` - polynomial whose roots have to be found.
#[allow(dead_code)]
fn init_simple<T>(poly: &Poly<T>) -> Vec<Complex<T>>
where
    T: Debug + Float + FloatConst + MulAdd<Output = T> + NumCast,
{
    // Convert degree from usize to float
    let n = poly.degree().unwrap_or(1);
    let n_f = T::from(n).unwrap();

    // Calculate the center of the circle.
    let a_n = poly.leading_coeff();
    let a_n_1 = poly[poly.len() - 2];
    let c = -a_n_1 / n_f / a_n;

    // Calculate the radius of the circle.
    let r = poly.eval(c).abs().powf(n_f.recip());

    // Pre-compute the constants of the exponent.
    let phi = T::one() * FloatConst::FRAC_PI_2() / n_f;
    let tau = (T::one() + T::one()) * FloatConst::PI();

    let initial: Vec<Complex<T>> = (1..=n)
        .map(|j| {
            let j_f = T::from(j).unwrap();
            let ex = tau * j_f / n_f + phi;
            let ex = Complex::i() * ex;
            ex.exp() * r + c
        })
        .collect();
    initial
}

/// Generate the initial approximation of the polynomial roots.
///
/// # Arguments
///
/// * `poly` - polynomial whose roots have to be found.
fn init<T>(poly: &Poly<T>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    // set = Vec<(k as usize, k as Float, ln(c_k) as Float)>
    let set: Vec<(usize, T, T)> = poly
        .coeffs
        .iter()
        .enumerate()
        .map(|(k, c)| (k, T::from(k).unwrap(), c.abs().ln()))
        .collect();

    // Convex hull
    // ch = Vec<(k as usize, k as Float)>
    let ch = convex_hull_top(&set);

    // r = Vec<(k_(i+1) - k_i as usize, r as Float)>
    let r: Vec<(usize, T)> = ch
        .windows(2)
        .map(|w| {
            // w[1] = k_(i+1), w[0] = k_i
            let tmp = (poly.coeffs[w[0].0] / poly.coeffs[w[1].0]).abs();
            (w[1].0 - w[0].0, tmp.powf((w[1].1 - w[0].1).recip()))
        })
        .collect();

    // Initial values
    let tau = (T::one() + T::one()) * FloatConst::PI();
    let initial: Vec<Complex<T>> = r
        .iter()
        .flat_map(|&(n_k, r)| {
            let n_k_f = T::from(n_k).unwrap();
            (0..n_k).map(move |i| {
                let i_f = T::from(i).unwrap();
                let ex = tau * i_f / n_k_f;
                (Complex::i() * ex).exp() * r
            })
        })
        .collect();
    initial
}

/// Calculate the upper convex hull of the given set of points.
///
/// # Arguments
///
/// * `set` - set of points.
fn convex_hull_top<T>(set: &[(usize, T, T)]) -> Vec<(usize, T)>
where
    T: Float,
{
    let mut stack = Vec::<(usize, T, T)>::new();
    stack.push(set[0]);
    stack.push(set[1]);

    for p in set.iter().skip(2) {
        loop {
            let length = stack.len();
            // There shall be at least 2 elements in the stack.
            if length < 2 {
                break;
            }
            let next_to_top = stack.get(length - 2).unwrap();
            let top = stack.last().unwrap();

            let c = cross_product((next_to_top.1, next_to_top.2), (top.1, top.2), (p.1, p.2));
            // Remove the top if it is not a strict turn to the right.
            if c < T::zero() {
                break;
            } else {
                stack.pop();
            }
        }
        stack.push(*p);
    }

    let res: Vec<_> = stack.iter().map(|&(a, b, _c)| (a, b)).collect();
    // It is be sorted by k.
    res
}

/// Compute the cross product of (p1 - p0) and (p2 - p0)
///
/// `(p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)`
fn cross_product<T>(p0: (T, T), p1: (T, T), p2: (T, T)) -> T
where
    T: Copy + Mul<Output = T> + Sub<Output = T>,
{
    let first = (p1.0 - p0.0, p1.1 - p0.1);
    let second = (p2.0 - p0.0, p2.1 - p0.1);
    first.0 * second.1 - second.0 * first.1
}

/// Implement read only indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// assert_eq!(2, p[2]);
/// ```
impl<T> Index<usize> for Poly<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.coeffs[i]
    }
}

/// Implement mutable indexing of polynomial returning its coefficients.
///
/// # Panics
///
/// Panics for out of bounds access.
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let mut p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// p[2] = 4;
/// assert_eq!(4, p[2]);
/// ```
impl<T> IndexMut<usize> for Poly<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.coeffs[i]
    }
}

/// Implementation of polynomial negation
impl<T: Copy + Neg<Output = T>> Neg for &Poly<T> {
    type Output = Poly<T>;

    fn neg(self) -> Self::Output {
        let c: Vec<_> = self.coeffs.iter().map(|&i| -i).collect();
        Poly { coeffs: c }
    }
}

/// Implementation of polynomial negation
impl<T: Copy + Neg<Output = T>> Neg for Poly<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for c in &mut self.coeffs {
            *c = Neg::neg(*c);
        }
        self
    }
}

/// Implementation of polynomial addition
impl<T: Copy + Num> Add for Poly<T> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.coeffs.iter().enumerate() {
                rhs[i] = rhs[i] + *c;
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i] + *c;
            }
            self
        };
        result.trim();
        result
    }
}

/// Implementation of polynomial addition
impl<T: Copy + Num> Add for &Poly<T> {
    type Output = Poly<T>;

    fn add(self, rhs: &Poly<T>) -> Poly<T> {
        let new_coeffs = utils::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), Add::add);
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and real number addition
impl<T: Add<Output = T> + Copy> Add<T> for Poly<T> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self[0] = self[0] + rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        self
    }
}

/// Implementation of polynomial and real number addition
impl<T: Add<Output = T> + Copy> Add<T> for &Poly<T> {
    type Output = Poly<T>;

    fn add(self, rhs: T) -> Self::Output {
        self.clone().add(rhs)
    }
}

macro_rules! impl_add_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Add<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn add(self, rhs: Poly<Self>) -> Poly<Self> {
                rhs + self
            }
        }
        $(#[$meta])*
        impl Add<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn add(self, rhs: &Poly<Self>) -> Poly<Self> {
                rhs + self
            }
        }
    };
}

impl_add_for_poly!(
    /// Implementation of f32 and polynomial addition
    f32
);
impl_add_for_poly!(
    /// Implementation of f64 and polynomial addition
    f64
);
impl_add_for_poly!(
    /// Implementation of i8 and polynomial addition
    i8
);
impl_add_for_poly!(
    /// Implementation of u8 and polynomial addition
    u8
);
impl_add_for_poly!(
    /// Implementation of i16 and polynomial addition
    i16
);
impl_add_for_poly!(
    /// Implementation of u16 and polynomial addition
    u16
);
impl_add_for_poly!(
    /// Implementation of i32 and polynomial addition
    i32
);
impl_add_for_poly!(
    /// Implementation of u32 and polynomial addition
    u32
);
impl_add_for_poly!(
    /// Implementation of i64 and polynomial addition
    i64
);
impl_add_for_poly!(
    /// Implementation of u64 and polynomial addition
    u64
);
impl_add_for_poly!(
    /// Implementation of i128 and polynomial addition
    i128
);
impl_add_for_poly!(
    /// Implementation of u128 and polynomial addition
    u128
);
impl_add_for_poly!(
    /// Implementation of isize and polynomial addition
    isize
);
impl_add_for_poly!(
    /// Implementation of usize and polynomial addition
    usize
);

/// Implementation of polynomial subtraction
impl<T: Copy + PartialEq + Sub<Output = T> + Zero> Sub for Poly<T> {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.len() < rhs.len() {
            // iterate on rhs and do the subtraction until self has values,
            // then invert the coefficients of rhs
            for i in 0..rhs.len() {
                rhs[i] = *self.coeffs.get(i).unwrap_or(&T::zero()) - rhs[i];
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i] - *c;
            }
            self
        };
        result.trim();
        result
    }
}

/// Implementation of polynomial subtraction
impl<T: Copy + PartialEq + Sub<Output = T> + Zero> Sub for &Poly<T> {
    type Output = Poly<T>;

    fn sub(self, rhs: Self) -> Poly<T> {
        let new_coeffs =
            utils::zip_longest_with(&self.coeffs, &rhs.coeffs, T::zero(), |x, y| x - y);
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial and real number subtraction
impl<T: Copy + Sub<Output = T>> Sub<T> for Poly<T> {
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self {
        self[0] = self[0] - rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        self
    }
}

/// Implementation of polynomial and real number subtraction
impl<T: Copy + Sub<Output = T>> Sub<T> for &Poly<T> {
    type Output = Poly<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.clone().sub(rhs)
    }
}

macro_rules! impl_sub_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Sub<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn sub(self, rhs: Poly<Self>) -> Poly<Self> {
                rhs.neg().add(self)
            }
        }
        $(#[$meta])*
        impl Sub<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn sub(self, rhs: &Poly<Self>) -> Poly<Self> {
                self.sub(rhs.clone())
            }
        }
    };
}

impl_sub_for_poly!(
    /// Implementation of f32 and polynomial subtraction
    f32
);
impl_sub_for_poly!(
    /// Implementation of f64 and polynomial subtraction
    f64
);
impl_sub_for_poly!(
    /// Implementation of i8 and polynomial subtraction
    i8
);
impl_sub_for_poly!(
    /// Implementation of i16 and polynomial subtraction
    i16
);
impl_sub_for_poly!(
    /// Implementation of i32 and polynomial subtraction
    i32
);
impl_sub_for_poly!(
    /// Implementation of i64 and polynomial subtraction
    i64
);
impl_sub_for_poly!(
    /// Implementation of i128 and polynomial subtraction
    i128
);
impl_sub_for_poly!(
    /// Implementation of isize and polynomial subtraction
    isize
);

/// Implementation of polynomial multiplication
impl<T: Copy + Mul<Output = T> + PartialEq + Zero> Mul for &Poly<T> {
    type Output = Poly<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Poly<T> {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_length = self.len() + rhs.len() - 1;
        let mut new_coeffs: Vec<T> = vec![T::zero(); new_length];
        for i in 0..self.len() {
            for j in 0..rhs.len() {
                let a = *self.coeffs.get(i).unwrap_or(&T::zero());
                let b = *rhs.coeffs.get(j).unwrap_or(&T::zero());
                let index = i + j;
                new_coeffs[index] = new_coeffs[index] + a * b;
            }
        }
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial multiplication
impl<T: Copy + Mul<Output = T> + PartialEq + Zero> Mul for Poly<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // Can't reuse arguments to avoid additional allocations.
        // The to arguments can't mutate during the loops.
        Mul::mul(&self, &rhs)
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Copy + Num> Mul<T> for Poly<T> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        if rhs.is_zero() {
            Self::zero()
        } else {
            for c in &mut self.coeffs {
                *c = *c * rhs;
            }
            self
        }
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Copy + Num> Mul<T> for &Poly<T> {
    type Output = Poly<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone().mul(rhs)
    }
}

macro_rules! impl_mul_for_poly {
    (
        $(#[$meta:meta])*
            $f:ty
    ) => {
        $(#[$meta])*
        impl Mul<Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn mul(self, rhs: Poly<Self>) -> Poly<Self> {
                rhs * self
            }
        }
        $(#[$meta])*
        impl Mul<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn mul(self, rhs: &Poly<Self>) -> Poly<Self> {
                rhs * self
            }
        }
    };
}

impl_mul_for_poly!(
    /// Implementation of f32 and polynomial multiplication
    f32
);
impl_mul_for_poly!(
    /// Implementation of f64 and polynomial multiplication
    f64
);
impl_mul_for_poly!(
    /// Implementation of i8 and polynomial multiplication
    i8
);
impl_mul_for_poly!(
    /// Implementation of u8 and polynomial multiplication
    u8
);
impl_mul_for_poly!(
    /// Implementation of i16 and polynomial multiplication
    i16
);
impl_mul_for_poly!(
    /// Implementation of u16 and polynomial multiplication
    u16
);
impl_mul_for_poly!(
    /// Implementation of i32 and polynomial multiplication
    i32
);
impl_mul_for_poly!(
    /// Implementation of u32 and polynomial multiplication
    u32
);
impl_mul_for_poly!(
    /// Implementation of i64 and polynomial multiplication
    i64
);
impl_mul_for_poly!(
    /// Implementation of u64 and polynomial multiplication
    u64
);
impl_mul_for_poly!(
    /// Implementation of i128 and polynomial multiplication
    i128
);
impl_mul_for_poly!(
    /// Implementation of u128 and polynomial multiplication
    u128
);
impl_mul_for_poly!(
    /// Implementation of isize and polynomial multiplication
    isize
);
impl_mul_for_poly!(
    /// Implementation of usize and polynomial multiplication
    usize
);

/// Implementation of polynomial and real number division
impl<T: Copy + Num> Div<T> for Poly<T> {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        for c in &mut self.coeffs {
            *c = *c / rhs;
        }
        self.trim();
        self
    }
}

/// Implementation of polynomial and real number division
impl<T: Copy + Num> Div<T> for &Poly<T> {
    type Output = Poly<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.clone().div(rhs)
    }
}

/// Implementation of division between polynomials
///
/// Panics
///
/// This method panics if the denominator is zero.
impl<T: Float> Div for &Poly<T> {
    type Output = Poly<T>;

    fn div(self, rhs: &Poly<T>) -> Self::Output {
        poly_div_impl(self.clone(), rhs).0
    }
}

/// Implementation of division between polynomials
///
/// Panics
///
/// This method panics if the denominator is zero.
impl<T: Float> Div for Poly<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        poly_div_impl(self, &rhs).0
    }
}

/// Implementation of reminder between polynomials.
///
/// Panics
///
/// This method panics if the denominator is zero.
impl<T: Float> Rem for &Poly<T> {
    type Output = Poly<T>;

    fn rem(self, rhs: &Poly<T>) -> Self::Output {
        poly_div_impl(self.clone(), rhs).1
    }
}

/// Implementation of reminder between polynomials.
///
/// Panics
///
/// This method panics if the denominator is zero.
impl<T: Float> Rem for Poly<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        poly_div_impl(self, &rhs).1
    }
}

/// Donald Ervin Knuth, The Art of Computer Programming: Seminumerical algorithms
/// Volume 2, third edition, section 4.6.1
/// Algorithm D: division of polynomials over a field.
///
/// Panics
///
/// This method panics if the denominator is zero.
#[allow(clippy::many_single_char_names)]
fn poly_div_impl<T: Float>(mut u: Poly<T>, v: &Poly<T>) -> (Poly<T>, Poly<T>) {
    let (m, n) = match (u.degree(), v.degree()) {
        (_, None) => panic!("Division by zero polynomial"),
        (None, _) => return (Poly::zero(), Poly::zero()),
        (Some(m), Some(n)) if m < n => return (Poly::zero(), u),
        (Some(m), Some(n)) => (m, n),
    };

    // 1/v_n
    let vn_rec = v.leading_coeff().recip();

    let mut q = Poly {
        coeffs: vec![T::zero(); m - n + 1],
    };

    for k in (0..=m - n).rev() {
        q[k] = u[n + k] * vn_rec;
        // n+k-1..=k
        for j in (k..n + k).rev() {
            u[j] = u[j] - q[k] * v[j - k];
        }
    }

    // (r_n-1, ..., r_0) = (u_n-1, ..., u_0)
    // reuse u coefficients.
    u.coeffs.truncate(n);
    // Trim take care of the case n=0.
    u.trim();
    // No need to trim q, its higher degree coefficient is always different from 0.
    (q, u)
}

/// Implementation of the additive identity for polynomials
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// use num_traits::Zero;
/// let zero = Poly::<u8>::zero();
/// assert!(zero.is_zero());
/// ```
impl<T: Copy + Num> Zero for Poly<T> {
    fn zero() -> Self {
        Self {
            coeffs: vec![T::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0] == T::zero()
    }
}

/// Implementation of the multiplicative identity for polynomials
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// use num_traits::One;
/// let one = Poly::<u8>::one();
/// assert!(one.is_one());
/// ```
impl<T: Copy + Num> One for Poly<T> {
    fn one() -> Self {
        Self {
            coeffs: vec![T::one()],
        }
    }

    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0] == T::one()
    }
}

impl<T: Float + FloatConst> Poly<T> {
    /// Polynomial multiplication through fast Fourier transform.
    ///
    /// # Arguments
    ///
    /// * `rhs` - right hand side of multiplication
    ///
    /// # Example
    ///
    /// ```
    /// use automatica::poly;
    /// let a = poly![1., 0., 3.];
    /// let b = poly![1., 0., 3.];
    /// let expected = &a * &b;
    /// let actual = a.mul_fft(b);
    /// assert_eq!(expected, actual);
    /// ```
    #[must_use]
    pub fn mul_fft(mut self, mut rhs: Self) -> Self {
        // Handle zero polynomial.
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        if self.is_one() {
            return rhs;
        } else if rhs.is_one() {
            return self;
        }
        // Both inputs shall have the same length.
        let res_degree = self.len() + rhs.len() - 1;
        self.extend(res_degree);
        rhs.extend(res_degree);
        // Convert the inputs into complex number vectors.
        let a: Vec<Complex<T>> = self
            .coeffs
            .iter()
            .map(|&x| std::convert::From::from(x))
            .collect();
        let b: Vec<Complex<T>> = rhs
            .coeffs
            .iter()
            .map(|&x| std::convert::From::from(x))
            .collect();
        // DFFT of the inputs.
        let a_fft = fft(a);
        let b_fft = fft(b);
        // Multiply the two transforms.
        let y_fft = utils::zip_with(&a_fft, &b_fft, |a, b| a * b).collect();
        // IFFT of the result.
        let y = ifft(y_fft);
        // Extract the real parts of the result.
        let coeffs = y.iter().map(|c| c.re).collect();
        let mut res = Self { coeffs };
        res.trim();
        res
    }
}

/// Integer logarithm of a power of two.
///
/// # Arguments
///
/// * `n` - power of two
fn log2(n: usize) -> usize {
    // core::mem::size_of::<usize>() * 8 - 1 - n.leading_zeros() as usize
    n.trailing_zeros() as usize
}

/// Reorder the elements of the vector using a bit inversion permutation.
///
/// # Arguments
///
/// * `a` - vector
/// * `bits` - number of lower bit on which the permutation shall act
#[allow(non_snake_case)]
fn bit_reverse_copy<T: Copy + Zero>(a: &[T], bits: usize) -> Vec<T> {
    let l = a.len();
    let mut A = vec![T::zero(); l];

    for k in 0..l {
        let r = rev(k, bits);
        *A.get_mut(r).unwrap() = *a.get(k).unwrap();
    }
    A
}

/// Reverse the last `l` bits of `k`.
///
/// # Arguments
///
/// * `k` - number on which the permutation acts.
/// * `l` - number of lower bits to reverse.
fn rev(k: usize, l: usize) -> usize {
    let mut r: usize = 0;
    for shift in 0..l {
        // Extract the "shift-th" bit.
        let bit = (k >> shift) & 1;
        // Push the bit to the back of the result.
        r = (r << 1) | bit;
    }
    r
}

/// Direct Fast Fourier Transform.
///
/// # Arguments
///
/// * `a` - vector
fn fft<T>(a: Vec<Complex<T>>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    iterative_fft(a, Transform::Direct)
}

/// Inverse Fast Fourier Transform.
///
/// # Arguments
///
/// * `y` - vector
fn ifft<T>(y: Vec<Complex<T>>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    iterative_fft(y, Transform::Inverse)
}

/// Extend the vector to a length that is the nex power of two.
///
/// # Arguments
///
/// * `a` - vector
fn extend_two_power_of_two<T: Copy + Zero>(mut a: Vec<T>) -> Vec<T> {
    let n = a.len();
    if n.is_power_of_two() {
        a
    } else {
        let pot = n.next_power_of_two();
        a.resize(pot, T::zero());
        a
    }
}

/// Type of Fourier transform.
#[derive(Clone, Copy)]
enum Transform {
    /// Direct fast Fourier transform.
    Direct,
    /// Inverse fast Fourier transform.
    Inverse,
}

/// Iterative fast Fourier transform algorithm.
/// T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, Introduction to Algorithms, 3rd edition, 2009
///
/// # Arguments
///
/// * `a` - input vector for the transform
/// * `dir` - transform "direction" (direct or inverse)
#[allow(clippy::many_single_char_names, non_snake_case)]
fn iterative_fft<T>(a: Vec<Complex<T>>, dir: Transform) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    let a = extend_two_power_of_two(a);
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let bits = log2(n);

    let mut A = bit_reverse_copy(&a, bits);

    let sign = match dir {
        Transform::Direct => T::one(),
        Transform::Inverse => -T::one(),
    };
    let tau = (T::one() + T::one()) * FloatConst::PI();

    for s in 1..=bits {
        let m = 1 << s;
        let m_f = T::from(m).unwrap();
        let exp = sign * tau / m_f;
        let w_n = Complex::from_polar(&T::one(), &exp);
        for k in (0..n).step_by(m) {
            let mut w = Complex::one();
            for j in 0..m / 2 {
                let t = A[k + j + m / 2] * w;
                let u = A[k + j];
                A[k + j] = u + t;
                A[k + j + m / 2] = u - t;
                w = w * w_n;
            }
        }
    }

    match dir {
        Transform::Direct => A,
        Transform::Inverse => {
            let n_f = T::from(n).unwrap();
            A.iter().map(|x| x / n_f).collect()
        }
    }
}

/// Implement printing of polynomial
///
/// # Example
/// ```
/// use automatica::polynomial::Poly;
/// let p = Poly::new_from_coeffs(&[0, 1, 2, 3]);
/// assert_eq!("+1*s +2*s^2 +3*s^3", format!("{}", p));
/// ```
impl<T: Display + One + PartialEq + Signed + Zero> Display for Poly<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        } else if self.len() == 1 {
            return write!(f, "{}", self.coeffs[0]);
        }
        let mut s = String::new();
        let mut sep = "";
        for (i, c) in self.coeffs.iter().enumerate() {
            // TODO use approx crate
            //if relative_eq!(*c, 0.0) {
            if *c == T::zero() {
                continue;
            }
            s.push_str(sep);
            #[allow(clippy::float_cmp)] // signum() returns either 1.0 or -1.0
            let sign = if c.signum() == T::one() { "+" } else { "" };
            if i == 0 {
                s.push_str(&format!("{}", c));
            } else if i == 1 {
                s.push_str(&format!("{}{}*s", sign, c));
            } else {
                s.push_str(&format!("{}{}*s^{}", sign, c, i));
            }
            sep = " ";
        }

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn poly_formatting() {
        assert_eq!("0", format!("{}", Poly::<i16>::zero()));
        assert_eq!("0", format!("{}", Poly::<i16>::new_from_coeffs(&[])));
        assert_eq!("1 +2*s^3 -4*s^4", format!("{}", poly!(1, 0, 0, 2, -4)));
    }

    #[test]
    fn poly_creation_coeffs() {
        let c = [4.3, 5.32];
        assert_eq!(c, Poly::new_from_coeffs(&c).coeffs.as_slice());

        let c2 = [0., 1., 1., 0., 0., 0.];
        assert_eq!([0., 1., 1.], Poly::new_from_coeffs(&c2).coeffs.as_slice());

        let zero: [f64; 1] = [0.];
        assert_eq!(zero, poly!(0., 0.).coeffs.as_slice());

        let int = [1, 2, 3, 4, 5];
        assert_eq!(int, Poly::new_from_coeffs(&int).coeffs.as_slice());

        let float = [0.1_f32, 0.34, 3.43];
        assert_eq!(float, Poly::new_from_coeffs(&float).coeffs.as_slice());
    }

    #[test]
    fn coeffs() {
        let int = [1, 2, 3, 4, 5];
        let p = Poly::new_from_coeffs(&int);
        assert_eq!(int, p.coeffs().as_slice());
    }

    #[test]
    fn poly_creation_roots() {
        assert_eq!(poly!(4., 4., 1.), Poly::new_from_roots(&[-2., -2.]));

        assert_eq!(poly!(4, 4, 1), Poly::new_from_roots(&[-2, -2]));

        assert!(vec![-2., -2.]
            .iter()
            .zip(
                Poly::new_from_roots(&[-2., -2.])
                    .real_roots()
                    .unwrap()
                    .iter()
            )
            .map(|(x, y): (&f64, &f64)| (x - y).abs())
            .all(|x| x < 0.000_001));

        assert!(vec![1.0_f32, 2., 3.]
            .iter()
            .zip(
                Poly::new_from_roots(&[1., 2., 3.])
                    .real_roots()
                    .unwrap()
                    .iter()
            )
            .map(|(x, y): (&f32, &f32)| (x - y).abs())
            .all(|x| x < 0.000_01));

        assert_eq!(
            poly!(0., -2., 1., 1.),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
    }

    #[test]
    fn len() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(3, p.len());
    }

    #[test]
    fn degree() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(Some(2), p.degree());

        let p2 = Poly::new_from_coeffs(&[0.]);
        assert_eq!(None, p2.degree());
    }

    #[test]
    fn extend_less() {
        let mut p1 = poly!(3, 4, 2);
        let p2 = p1.clone();
        p1.extend(1);
        assert_eq!(p1, p2);
    }

    #[test]
    fn extend_more() {
        let mut p1 = poly!(3, 4, 2);
        let p2 = Poly {
            coeffs: vec![3, 4, 2, 0, 0, 0, 0],
        };
        p1.extend(6);
        assert_eq!(p1, p2);
    }

    #[test]
    fn extend_zero() {
        let mut p1 = Poly::<u32>::zero();
        let p2 = Poly {
            coeffs: vec![0, 0, 0, 0],
        };
        p1.extend(3);
        assert_eq!(p1, p2);
    }

    #[test]
    fn poly_eval() {
        let p = poly!(1., 2., 3.);
        assert_abs_diff_eq!(86., p.eval(5.), epsilon = 0.);

        assert_abs_diff_eq!(0., Poly::<f64>::zero().eval(6.4), epsilon = 0.);

        let p2 = poly!(3, 4, 1);
        assert_eq!(143, p2.eval(10));
    }

    #[test]
    fn poly_cmplx_eval() {
        let p = poly!(1., 1., 1.);
        let c = Complex::new(1.0, 1.0);
        let res = Complex::new(2.0, 3.0);
        assert_eq!(res, p.eval(c));

        assert_eq!(
            Complex::zero(),
            Poly::<f64>::new_from_coeffs(&[]).eval(Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_neg() {
        let p1 = poly!(1., 2.34, -4.2229);
        let p2 = -&p1;
        assert_eq!(p1, -p2);
    }

    #[test]
    fn poly_add() {
        assert_eq!(poly!(4., 4., 4.), poly!(1., 2., 3.) + poly!(3., 2., 1.));

        assert_eq!(poly!(4., 4., 3.), poly!(1., 2., 3.) + poly!(3., 2.));

        assert_eq!(poly!(4., 4., 1.), poly!(1., 2.) + poly!(3., 2., 1.));

        assert_eq!(poly!(4., 4.), poly!(1., 2., 3.) + poly!(3., 2., -3.));

        assert_eq!(poly!(-2., 2., 3.), poly!(1., 2., 3.) + -3.);

        assert_eq!(poly!(0, 2, 3), 2 + poly!(1, 2, 3) + -3);

        assert_eq!(poly!(9.0_f32, 2., 3.), 3. + poly!(1.0_f32, 2., 3.) + 5.);

        let p = poly!(-2, 2, 3);
        let p2 = &p + &p;
        let p3 = &p2 + &p;
        assert_eq!(poly!(-6, 6, 9), p3);
    }

    #[test]
    fn poly_add_real_number() {
        assert_eq!(poly!(5, 4, 3), 1 + &poly!(4, 4, 3));
        assert_eq!(poly!(6, 4, 3), &poly!(5, 4, 3) + 1);
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn poly_sub() {
        assert_eq!(poly!(-2., 0., 2.), poly!(1., 2., 3.) - poly!(3., 2., 1.));

        assert_eq!(poly!(-2., 0., 3.), poly!(1., 2., 3.) - poly!(3., 2.));

        assert_eq!(poly!(-2., 0., -1.), poly!(1., 2.) - poly!(3., 2., 1.));

        assert_eq!(poly!(-2., 0., 6.), poly!(1., 2., 3.) - poly!(3., 2., -3.));

        let p = poly!(1., 1.);
        assert_eq!(Poly::zero(), &p - &p);

        assert_eq!(poly!(-10., 1.), poly!(2., 1.) - 12.);

        assert_eq!(poly!(-1., -1.), 1. - poly!(2., 1.));

        assert_eq!(poly!(-1_i8, -1), 1_i8 - poly!(2, 1));

        assert_eq!(poly!(-10, 1), poly!(2, 1) - 12);
    }

    #[test]
    fn poly_sub_real_number() {
        assert_eq!(poly!(-3, -4, -3), 1 - &poly!(4, 4, 3));
        assert_eq!(poly!(4, 4, 3), &poly!(5, 4, 3) - 1);
    }

    #[test]
    #[should_panic]
    fn poly_sub_panic() {
        let p = poly!(1, 2, 3) - 3_u32;
        // The assert is used only to avoid code optimization in release mode.
        assert_eq!(p.coeffs, vec![]);
    }

    #[test]
    #[allow(clippy::erasing_op)]
    fn poly_mul() {
        assert_eq!(
            poly!(0., 0., -1., 0., -1.),
            poly!(1., 0., 1.) * poly!(0., 0., -1.)
        );

        assert_eq!(Poly::zero(), poly!(1., 0., 1.) * Poly::zero());

        assert_eq!(poly!(1., 0., 1.), poly!(1., 0., 1.) * Poly::one());

        assert_eq!(poly!(-3., 0., -3.), poly!(1., 0., 1.) * poly!(-3.));

        let p = poly!(-3., 0., -3.);
        assert_eq!(poly!(9., 0., 18., 0., 9.), &p * &p);

        assert_eq!(
            poly!(-266.07_f32, 0., -266.07),
            4.9 * poly!(1.0_f32, 0., 1.) * -54.3
        );

        assert_eq!(Poly::zero(), 0. * poly!(1., 0., 1.));

        assert_eq!(Poly::zero(), poly!(1, 0, 1) * 0);

        assert_eq!(Poly::zero(), &poly!(1, 0, 1) * 0);

        assert_eq!(poly!(3, 0, 3), &poly!(1, 0, 1) * 3);
    }

    #[test]
    fn poly_mul_real_number() {
        assert_eq!(poly!(4, 4, 3), 1 * &poly!(4, 4, 3));
        assert_eq!(poly!(10, 8, 6), &poly!(5, 4, 3) * 2);
    }

    #[test]
    fn poly_div() {
        assert_eq!(poly!(0.5, 0., 0.5), poly!(1., 0., 1.) / 2.0);

        assert_eq!(poly!(4, 0, 5), poly!(8, 1, 11) / 2);

        let inf = std::f32::INFINITY;
        assert_eq!(Poly::zero(), poly!(1., 0., 1.) / inf);

        assert_eq!(poly!(inf, -inf, inf), poly!(1., -2.3, 1.) / 0.);
    }

    #[test]
    fn poly_mutable_div() {
        let mut p = poly!(3, 4, 5);
        p.div_mut(2);
        assert_eq!(poly!(1, 2, 2), p);
    }

    #[test]
    #[should_panic]
    fn div_panic() {
        let _ = poly_div_impl(poly!(6., 5., 1.), &poly!(0.));
    }

    #[test]
    fn poly_division_impl() {
        let d1 = poly_div_impl(poly!(6., 5., 1.), &poly!(2., 1.));
        assert_eq!(poly!(3., 1.), d1.0);
        assert_eq!(poly!(0.), d1.1);

        let d2 = poly_div_impl(poly!(5., 3., 1.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.5), d2.0);
        assert_eq!(poly!(3.), d2.1);

        let d3 = poly_div_impl(poly!(3., 1.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.), d3.0);
        assert_eq!(poly!(3., 1.), d3.1);

        let d4 = poly_div_impl(poly!(0.), &poly!(4., 6., 2.));
        assert_eq!(poly!(0.), d4.0);
        assert_eq!(poly!(0.), d4.1);

        let d5 = poly_div_impl(poly!(4., 6., 2.), &poly!(2.));
        assert_eq!(poly!(2., 3., 1.), d5.0);
        assert_eq!(poly!(0.), d5.1);
    }

    #[test]
    fn two_poly_div() {
        let q = poly!(-1., 0., 0., 0., 1.) / poly!(1., 0., 1.);
        assert_eq!(poly!(-1., 0., 1.), q);
    }

    #[test]
    fn two_poly_div_ref() {
        let q = &poly!(-1., 0., 0., 0., 1.) / &poly!(1., 0., 1.);
        assert_eq!(poly!(-1., 0., 1.), q);
    }

    #[test]
    fn two_poly_rem() {
        let r = poly!(-4., 0., -2., 1.) % poly!(-3., 1.);
        assert_eq!(poly!(5.), r);
    }

    #[test]
    fn two_poly_rem_ref() {
        let r = &poly!(-4., 0., -2., 1.) % &poly!(-3., 1.);
        assert_eq!(poly!(5.), r);
    }

    #[test]
    fn indexing() {
        assert_abs_diff_eq!(3., poly!(1., 3.)[1], epsilon = 0.);

        let mut p = Poly::new_from_roots(&[1., 4., 5.]);
        p[2] = 3.;
        assert_eq!(poly!(-20., 29., 3., 1.), p);
    }

    #[test]
    fn derive() {
        let p = poly!(1_u8, 2, 4, 8, 16);
        let p_prime = poly!(2_u8, 8, 24, 64);
        assert_eq!(p_prime, p.derive());
    }

    #[test]
    fn integrate() {
        let p = poly!(1_u8, 2, 4, 8, 16);
        let p2 = poly!(9_u8, 1, 1, 1, 2, 3);
        // Integer division.
        assert_eq!(p2, p.integrate(9));
    }

    #[test]
    fn derive_integrate() {
        let d = poly!(1.3, 3.5, -2.3, -1.6);
        let i = d.integrate(3.2);
        assert_eq!(d, i.derive());
    }

    #[test]
    fn float_coeffs_identities() {
        assert!(Poly::<f64>::zero().is_zero());
        assert!(Poly::<f64>::one().is_one());

        assert!(Poly::<f32>::zero().is_zero());
        assert!(Poly::<f32>::one().is_one());
    }

    #[test]
    fn integer_coeffs_identities() {
        assert!(Poly::<i8>::zero().is_zero());
        assert!(Poly::<i8>::one().is_one());

        assert!(Poly::<u8>::zero().is_zero());
        assert!(Poly::<u8>::one().is_one());

        assert!(Poly::<i16>::zero().is_zero());
        assert!(Poly::<i16>::one().is_one());

        assert!(Poly::<u16>::zero().is_zero());
        assert!(Poly::<u16>::one().is_one());

        assert!(Poly::<i32>::zero().is_zero());
        assert!(Poly::<i32>::one().is_one());

        assert!(Poly::<u32>::zero().is_zero());
        assert!(Poly::<u32>::one().is_one());

        assert!(Poly::<i64>::zero().is_zero());
        assert!(Poly::<i64>::one().is_one());

        assert!(Poly::<u64>::zero().is_zero());
        assert!(Poly::<u64>::one().is_one());

        assert!(Poly::<i128>::zero().is_zero());
        assert!(Poly::<i128>::one().is_one());

        assert!(Poly::<u128>::zero().is_zero());
        assert!(Poly::<u128>::one().is_one());

        assert!(Poly::<isize>::zero().is_zero());
        assert!(Poly::<isize>::one().is_one());

        assert!(Poly::<usize>::zero().is_zero());
        assert!(Poly::<usize>::one().is_one());
    }

    #[quickcheck]
    fn leading_coefficient(c: f32) -> bool {
        relative_eq!(c, poly!(1., -5., c).leading_coeff())
    }

    #[test]
    fn monic_poly() {
        let p = poly!(-3., 6., 9.);
        let (p2, c) = p.monic();
        assert_relative_eq!(9., c);
        assert_relative_eq!(1., p2.leading_coeff());
    }

    #[test]
    fn monic_mutable_poly() {
        let mut p = poly!(-3., 6., 9.);
        let c = p.monic_mut();
        assert_relative_eq!(9., c);
        assert_relative_eq!(1., p.leading_coeff());
    }

    #[test]
    fn failing_companion() {
        let p = Poly::<f32>::zero();
        assert_eq!(None, p.companion());
    }
}

mod compile_fail_test {
    /// ```compile_fail
    /// use automatica::{poly, Eval};
    /// let p = poly!(1.0e200, 2., 3.);
    /// p.eval(5.0_f32);
    /// ```
    #[allow(dead_code)]
    fn a() {}

    /// ``` compile_fail
    /// use automatica::{poly, Eval};
    /// let p = poly!(1.5, 2., 3.);
    /// assert_eq!(86, p.eval(5));
    /// ```
    #[allow(dead_code)]
    fn b() {}
}

#[cfg(test)]
mod tests_fft {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn reverse_bit() {
        assert_eq!(0, rev(0, 3));
        assert_eq!(4, rev(1, 3));
        assert_eq!(2, rev(2, 3));
        assert_eq!(6, rev(3, 3));
        assert_eq!(1, rev(4, 3));
        assert_eq!(5, rev(5, 3));
        assert_eq!(3, rev(6, 3));
        assert_eq!(7, rev(7, 3));
    }

    #[test]
    fn reverse_copy() {
        let a = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let l = log2(a.len());
        let b = bit_reverse_copy(&a, l);
        let a = vec![0, 4, 2, 6, 1, 5, 3, 7];
        assert_eq!(a, b);
    }

    #[test]
    fn fft_iterative() {
        let one = Complex::one();
        let a = vec![one * 1., one * 0., one * 1.];
        // `a` is extended to for elements
        let f = iterative_fft(a, Transform::Direct);
        let expected = vec![one * 2., one * 0., one * 2., one * 0.];
        assert_eq!(expected, f);
    }

    #[test]
    fn fft_ifft() {
        let one = Complex::one();
        let a = vec![one * 1., one * 0., one * 1., one * 0.];
        let f = fft(a.clone());
        let a2 = ifft(f);
        assert_eq!(a, a2);
    }

    #[test]
    fn multiply_fft() {
        let a = poly![1., 0., 3.];
        let b = poly![1., 0., 3.];
        let expected = &a * &b;
        let actual = a.mul_fft(b);
        assert_eq!(expected, actual);
    }

    #[test]
    fn multiply_fft_one() {
        let a = poly![1., 0., 3.];
        let b = Poly::one();
        let actual = a.clone().mul_fft(b);
        assert_eq!(a, actual);

        let c = Poly::one();
        let d = poly![1., 0., 3.];
        let actual = c.mul_fft(d.clone());
        assert_eq!(d, actual);
    }

    #[test]
    fn multiply_fft_zero() {
        let a = poly![1., 0., 3.];
        let b = Poly::zero();
        let actual = a.mul_fft(b);
        assert_eq!(Poly::zero(), actual);
    }
}

#[cfg(test)]
mod tests_roots {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn quadratic_roots_with_real_values() {
        let root1 = -1.;
        let root2 = -2.;
        assert_eq!(Some((root1, root2)), real_quadratic_roots(3., 2.));

        assert_eq!(None, real_quadratic_roots(-6., 10.));

        let root3 = 3.;
        assert_eq!(Some((root3, root3)), real_quadratic_roots(-6., 9.));
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
        let roots = &[-1., 1., 0.];
        let p = Poly::new_from_roots(roots);
        assert_eq!(roots, p.real_roots().unwrap().as_slice());
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
        assert_eq!((root1, root2), complex_quadratic_roots(3., 2.));

        let root1 = Complex::<f64>::new(-0., -1.);
        let root2 = Complex::<f64>::new(-0., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots(0., 1.));

        let root1 = Complex::<f64>::new(3., -1.);
        let root2 = Complex::<f64>::new(3., 1.);
        assert_eq!((root1, root2), complex_quadratic_roots(-6., 10.));

        let root1 = Complex::<f64>::new(3., 0.);
        assert_eq!((root1, root1), complex_quadratic_roots(-6., 9.));
    }

    #[test]
    fn iterative_roots_finder() {
        let roots = &[10.0_f32, 10. / 323.4, 1., -2., 3.];
        let poly = Poly::new_from_roots(roots);
        let rf = RootsFinder::new(poly);
        let actual = rf.roots_finder();
        assert_eq!(roots.len(), actual.len());
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
}
