use std::ops::{Add, Mul, Sub};

use nalgebra::{DMatrix, DVector, Schur};
use num_complex::{Complex, Complex64};
use num_traits::Zero;

/// Polynomial object
///
/// Contains the vector of coefficients form the lowest to the higher degree
///
/// p(x) = c0 + c1*x + c2*x^2 + ...
#[derive(Debug, PartialEq, Clone)]
pub struct Poly {
    coeffs: Vec<f64>,
}

/// Implementation methods for Poly struct
impl Poly {
    /// Create a new polynomial given a slice of real coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    pub fn new_from_coeffs(coeffs: &[f64]) -> Self {
        Poly {
            coeffs: Poly::trim(coeffs).into(),
        }
    }

    /// Create a new polynomial given a slice of real roots
    ///
    /// # Arguments
    ///
    /// * `roots` - slice of roots
    pub fn new_from_roots(roots: &[f64]) -> Self {
        roots.iter().fold(Poly::new_from_coeffs(&[1.]), |acc, &r| {
            acc * Poly::new_from_coeffs(&[-r, 1.])
        })
    }

    /// Trim the zeros coefficients of high degree terms
    ///
    /// # Arguments
    ///
    /// * `coeffs` - slice of coefficients
    fn trim(coeffs: &[f64]) -> &[f64] {
        if let Some(p) = coeffs.iter().rposition(|&c| c != 0.0) {
            &coeffs[..=p]
        } else if coeffs.iter().any(|&c| c == 0.0) {
            // Case where all coefficients are zero.
            &coeffs[0..0]
        } else {
            &coeffs
        }
    }

    /// Degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Vector of the polynomial's coefficients
    pub fn coeffs(&self) -> Vec<f64> {
        self.coeffs.clone()
    }

    /// Build the companion matrix of the polynomial.
    ///
    /// Subdiagonal terms are 1., rightmost column contains the coefficients
    pub fn companion(&self) -> DMatrix<f64> {
        let length = self.degree();
        let hi_coeff = self.coeffs[length];
        DMatrix::from_fn(length, length, |i, j| {
            if j == length - 1 {
                -self.coeffs[i] / hi_coeff
            } else if i == j + 1 {
                1.
            } else {
                0.
            }
        })
    }

    /// Find the roots of the polynomial
    pub fn roots(&self) -> DVector<f64> {
        // Build the companion matrix
        let comp = self.companion();
        let schur = Schur::new(comp);
        schur.eigenvalues().unwrap()
    }
}

/// Trait for the implementation of polynomial evaluation
trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: T) -> T;
}

impl Eval<Complex64> for Poly {
    fn eval(&self, x: Complex64) -> Complex64 {
        self.coeffs
            .iter()
            .rev()
            .fold(Complex::zero(), |acc, &c| acc * x + c)
    }
}

impl Eval<f64> for Poly {
    fn eval(&self, x: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }
}

/// Implementation of polynomial addition
impl Add for Poly {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Check which polynomial has the highest degree
        let new_coeffs = if self.degree() < other.degree() {
            let mut res = other.coeffs.to_vec();
            for (i, c) in self.coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else if other.degree() < self.degree() {
            let mut res = self.coeffs.to_owned();
            for (i, c) in other.coeffs.iter().enumerate() {
                res[i] += c;
            }
            res
        } else {
            zip_with(&self.coeffs, &other.coeffs, |l, r| l + r)
        };
        Poly::new_from_coeffs(&new_coeffs)
    }
}

/// Implementation of polynomial subtraction
impl Sub for Poly {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // Just multiply 'other' by -1 and use addition.
        let sub_p: Vec<_> = other.coeffs.iter().map(|&c| -c).collect();
        self.add(Poly::new_from_coeffs(&sub_p))
    }
}

/// Implementation of polynomial multiplication
impl Mul for Poly {
    type Output = Poly;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Self) -> Self {
        // Polynomial multiplication is implemented as discrete convolution.
        let new_degree = self.degree() + other.degree();
        let mut new_coeffs: Vec<f64> = vec![0.; new_degree + 1];
        for i in 0..=self.degree() {
            for j in 0..=other.degree() {
                let a = self.coeffs.get(i).unwrap_or(&0.);
                let b = other.coeffs.get(j).unwrap_or(&0.);
                new_coeffs[i + j] += a * b;
            }
        }
        Poly::new_from_coeffs(&new_coeffs)
    }
}

// fn zipWith<U, C>(combo: C, left: U, right: U) -> impl Iterator
// where
//     U: Iterator,
//     C: FnMut(U::Item, U::Item) -> U::Item,
// {
//     left.zip(right).map(move |(l, r)| combo(l, r))
// }
/// Zip two slices with the given function
///
/// # Arguments
///
/// * `left` - first slice to zip
/// * `right` - second slice to zip
/// * `f` - function used to zip the two lists
fn zip_with<T, F>(left: &[T], right: &[T], mut f: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> T,
{
    left.iter().zip(right).map(|(l, r)| f(l, r)).collect()
}

/// Evaluate rational function at x
///
/// # Arguments
///
/// * `x` - Value for the evaluation
/// * `num` - Coefficients of the numerator polynomial. First element is the higher order coefficient
/// * `denom` - Coefficients of the denominator polynomial. First element is the higher order coefficient
// pub fn ratevl<T>(x: T, num: &[T], denom: &[T]) -> T
// where
//     T: Float,
// {
//     if x <= T::one() {
//         polynom_eval(x, num) / polynom_eval(x, denom)
//     } else {
//         // To avoid overflow the result is the same if coefficients are
//         // reversed and evaluated at 1/x
//         let z = x.recip();
//         polynom_eval_rev(z, num) / polynom_eval_rev(z, denom)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn polynom_ratevl_test() {
    //     let num = [1.0, 4.0, 3.0];
    //     let den = [2.0, -6.5, 0.4];
    //     let ratio1 = ratevl(3.3, &num, &den);
    //     assert_eq!(ratio1, 37.10958904109594);

    //     let num2 = [3.0, 4.0, 1.0];
    //     let den2 = [0.4, -6.5, 2.0];
    //     let ratio2 = ratevl(1.0 / 3.3, &num2, &den2);
    //     assert_eq!(ratio2, 37.10958904109594);

    //     assert_eq!(ratio1, ratio2);
    // }

    #[test]
    fn poly_creation_coeffs_test() {
        let c = [4.3, 5.32];
        assert_eq!(c, Poly::new_from_coeffs(&c).coeffs.as_slice());

        let c2 = [0., 1., 1., 0., 0., 0.];
        assert_eq!([0., 1., 1.], Poly::new_from_coeffs(&c2).coeffs.as_slice());

        let empty: [f64; 0] = [];
        assert_eq!(empty, Poly::new_from_coeffs(&[0., 0.]).coeffs.as_slice());
    }

    #[test]
    fn poly_creation_roots_test() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_roots(&[-2., -2.])
        );

        assert!(
            (DVector::from_element(2, -2.) - Poly::new_from_roots(&[-2., -2.]).roots())
                .abs()
                .iter()
                .all(|&x| x < 0.000001)
        );

        assert!((DVector::from_column_slice(&[1., 2., 3.])
            - Poly::new_from_roots(&[1., 2., 3.]).roots())
        .abs()
        .iter()
        .all(|&x| x < 0.000001));

        assert_eq!(
            Poly::new_from_coeffs(&[0., -2., 1., 1.]),
            Poly::new_from_roots(&[-0., -2., 1.])
        );
    }

    #[test]
    fn poly_f64_eval_test() {
        let p = Poly::new_from_coeffs(&[1., 2., 3.]);
        assert_eq!(86., p.eval(5.));

        assert_eq!(0.0, Poly::new_from_coeffs(&[]).eval(6.4));
    }

    #[test]
    fn poly_cmplx_eval_test() {
        let p = Poly::new_from_coeffs(&[1., 1., 1.]);
        let c = Complex::new(1.0, 1.0);
        let res = Complex::new(2.0, 3.0);
        assert_eq!(res, p.eval(c));

        assert_eq!(
            Complex::zero(),
            Poly::new_from_coeffs(&[]).eval(Complex::new(2., 3.))
        );
    }

    #[test]
    fn poly_add_test() {
        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4., 1.]),
            Poly::new_from_coeffs(&[1., 2.,]) + Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[4., 4.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) + Poly::new_from_coeffs(&[3., 2., -3.])
        );
    }

    #[test]
    fn poly_sub_test() {
        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 2.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 3.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., -1.]),
            Poly::new_from_coeffs(&[1., 2.,]) - Poly::new_from_coeffs(&[3., 2., 1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-2., 0., 6.]),
            Poly::new_from_coeffs(&[1., 2., 3.,]) - Poly::new_from_coeffs(&[3., 2., -3.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[]),
            Poly::new_from_coeffs(&[1., 1.]) - Poly::new_from_coeffs(&[1., 1.])
        );
    }

    #[test]
    fn poly_mul_test() {
        assert_eq!(
            Poly::new_from_coeffs(&[0., 0., -1., 0., -1.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[0., 0., -1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[0.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[0.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[1., 0., 1.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[1.])
        );

        assert_eq!(
            Poly::new_from_coeffs(&[-3., 0., -3.]),
            Poly::new_from_coeffs(&[1., 0., 1.]) * Poly::new_from_coeffs(&[-3.])
        );
    }
}
