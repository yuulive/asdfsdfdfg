//! Arithmetic module for polynomials
use num_complex::Complex;
use num_traits::{Float, FloatConst, Num, One, Zero};

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::{
    iterator,
    polynomial::{fft, Poly},
};

/// Implementation of polynomial negation
impl<T: Clone + Neg<Output = T>> Neg for &Poly<T> {
    type Output = Poly<T>;

    fn neg(self) -> Self::Output {
        let c: Vec<_> = self.coeffs.iter().map(|i| -i.clone()).collect();
        // The polynomial cannot be empty.
        Poly { coeffs: c }
    }
}

/// Implementation of polynomial negation
impl<T: Clone + Neg<Output = T>> Neg for Poly<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for c in &mut self.coeffs {
            *c = Neg::neg(c.clone());
        }
        // The polynomial cannot be empty.
        self
    }
}

/// Implementation of polynomial addition
impl<T: Add<Output = T> + Clone + PartialEq + Zero> Add for Poly<T> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.coeffs.iter().enumerate() {
                rhs[i] = rhs[i].clone() + c.clone();
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i].clone() + c.clone();
            }
            self
        };
        result.trim();
        // The polynomial cannot be empty, trim has already the postcondition.
        result
    }
}

/// Implementation of polynomial addition
impl<'a, T> Add<&'a Poly<T>> for Poly<T>
where
    T: Add<&'a T, Output = T> + Clone + PartialEq + Zero,
{
    type Output = Self;

    fn add(mut self, rhs: &'a Poly<T>) -> Self {
        let mut result = if self.degree() < rhs.degree() {
            for (i, c) in self.coeffs.iter_mut().enumerate() {
                *c = c.clone() + &rhs[i];
            }
            let l = self.len();
            self.coeffs.extend_from_slice(&rhs.coeffs[l..]);
            self
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i].clone() + c;
            }
            self
        };
        result.trim();
        // The polynomial cannot be empty, trim has already the postcondition.
        result
    }
}

/// Implementation of polynomial addition
impl<T: Clone + PartialEq + Zero> Add for &Poly<T> {
    type Output = Poly<T>;

    fn add(self, rhs: &Poly<T>) -> Poly<T> {
        let zero = T::zero();
        let result = iterator::zip_longest_with(&self.coeffs, &rhs.coeffs, &zero, |x, y| {
            x.clone() + y.clone()
        });
        // The polynomial cannot be empty.
        Poly::new_from_coeffs_iter(result)
    }
}

/// Implementation of polynomial and real number addition
impl<T: Add<Output = T> + Clone> Add<T> for Poly<T> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self {
        self[0] = self[0].clone() + rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        // The polynomial cannot be empty.
        self
    }
}

/// Implementation of polynomial and real number addition
impl<'a, T: Add<&'a T, Output = T> + Clone> Add<&'a T> for Poly<T> {
    type Output = Self;

    fn add(mut self, rhs: &'a T) -> Self {
        self[0] = self[0].clone() + rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        // The polynomial cannot be empty.
        self
    }
}

/// Implementation of polynomial and real number addition
impl<T: Add<Output = T> + Clone> Add<T> for &Poly<T> {
    type Output = Poly<T>;

    fn add(self, rhs: T) -> Self::Output {
        // The polynomial cannot be empty.
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
                // The polynomial cannot be empty.
                rhs + self
            }
        }
        $(#[$meta])*
        impl Add<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn add(self, rhs: &Poly<Self>) -> Poly<Self> {
                // The polynomial cannot be empty.
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
impl<T: Clone + PartialEq + Sub<Output = T> + Zero> Sub for Poly<T> {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self {
        // Check which polynomial has the highest degree.
        // Mutate the arguments since are passed as values.
        let mut result = if self.len() < rhs.len() {
            // iterate on rhs and do the subtraction until self has values,
            // then invert the coefficients of rhs
            for (i, c) in rhs.coeffs.iter_mut().enumerate() {
                *c = self.coeffs.get(i).unwrap_or(&T::zero()).clone() - c.clone();
            }
            rhs
        } else {
            for (i, c) in rhs.coeffs.iter().enumerate() {
                self[i] = self[i].clone() - c.clone();
            }
            self
        };
        result.trim();
        // The polynomial cannot be empty, trim has already the postcondition.
        result
    }
}

/// Implementation of polynomial subtraction
impl<T: Clone + PartialEq + Sub<Output = T> + Zero> Sub for &Poly<T> {
    type Output = Poly<T>;

    fn sub(self, rhs: Self) -> Poly<T> {
        let zero = T::zero();
        let result = iterator::zip_longest_with(&self.coeffs, &rhs.coeffs, &zero, |x, y| {
            x.clone() - y.clone()
        });
        // The polynomial cannot be empty.
        Poly::new_from_coeffs_iter(result)
    }
}

/// Implementation of polynomial and real number subtraction
impl<T: Clone + Sub<Output = T>> Sub<T> for Poly<T> {
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self {
        self[0] = self[0].clone() - rhs;
        // Non need for trimming since the addition of a float doesn't
        // modify the coefficients of order higher than zero.
        // The polynomial cannot be empty.
        self
    }
}

/// Implementation of polynomial and real number subtraction
impl<T: Clone + Sub<Output = T>> Sub<T> for &Poly<T> {
    type Output = Poly<T>;

    fn sub(self, rhs: T) -> Self::Output {
        // The polynomial cannot be empty.
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
        // The polynomial cannot be empty.
                rhs.neg().add(self)
            }
        }
        $(#[$meta])*
        impl Sub<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn sub(self, rhs: &Poly<Self>) -> Poly<Self> {
        // The polynomial cannot be empty.
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
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Mul for &Poly<T> {
    type Output = Poly<T>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Poly<T> {
        // Shortcut if one of the factors is zero.
        if self.is_zero() || rhs.is_zero() {
            return Poly::zero();
        }
        // Polynomial multiplication is implemented as discrete convolution.
        let new_length = self.len() + rhs.len() - 1;
        debug_assert!(new_length > 0);
        let mut new_coeffs: Vec<T> = vec![T::zero(); new_length];
        for i in 0..self.len() {
            for j in 0..rhs.len() {
                let a = self.coeffs[i].clone();
                let b = rhs.coeffs[j].clone();
                let index = i + j;
                new_coeffs[index] = new_coeffs[index].clone() + a * b;
            }
        }
        // The number of coefficients is at least one.
        // No need to trim since the last coefficient is not zero.
        Poly { coeffs: new_coeffs }
    }
}

/// Implementation of polynomial multiplication
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Mul for Poly<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // Can't reuse arguments to avoid additional allocations.
        // The two arguments can't mutate during the loops.
        Mul::mul(&self, &rhs)
    }
}

/// Implementation of polynomial multiplication
impl<T: Clone + Mul<Output = T> + PartialEq + Zero> Mul<&Poly<T>> for Poly<T> {
    type Output = Self;

    fn mul(self, rhs: &Poly<T>) -> Self {
        // Can't reuse arguments to avoid additional allocations.
        // The two arguments can't mutate during the loops.
        Mul::mul(&self, rhs)
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Clone + Num> Mul<T> for Poly<T> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        if rhs.is_zero() {
            Self::zero()
        } else {
            for c in &mut self.coeffs {
                *c = c.clone() * rhs.clone();
            }
            // The polynomial cannot be empty.
            self
        }
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Clone + Num> Mul<&T> for Poly<T> {
    type Output = Self;

    fn mul(mut self, rhs: &T) -> Self {
        if rhs.is_zero() {
            Self::zero()
        } else {
            for c in &mut self.coeffs {
                *c = c.clone() * rhs.clone();
            }
            // The polynomial cannot be empty.
            self
        }
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Clone + Num> Mul<T> for &Poly<T> {
    type Output = Poly<T>;

    fn mul(self, rhs: T) -> Self::Output {
        // The polynomial cannot be empty.
        self.clone().mul(rhs)
    }
}

/// Implementation of polynomial and float multiplication
impl<T: Clone + Num> Mul<&T> for &Poly<T> {
    type Output = Poly<T>;

    fn mul(self, rhs: &T) -> Self::Output {
        // The polynomial cannot be empty.
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
                // The polynomial cannot be empty.
                rhs * self
            }
        }
        $(#[$meta])*
        impl Mul<&Poly<$f>> for $f {
            type Output = Poly<Self>;

            fn mul(self, rhs: &Poly<Self>) -> Poly<Self> {
                // The polynomial cannot be empty.
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
        let res_length = self.len() + rhs.len() - 1;
        let res_degree = res_length - 1;
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
        let a_fft = fft::fft(a);
        let b_fft = fft::fft(b);
        // Multiply the two transforms.
        let y_fft = iterator::zip_with(&a_fft, &b_fft, |a, b| a * b).collect();
        // IFFT of the result.
        let y = fft::ifft(y_fft);
        // Extract the real parts of the result.
        // Keep the first res_length elements since is the number of coefficients
        // of the result.
        // No need to trim since the last coefficient cannot be zero.
        let coeffs = y.iter().map(|c| c.re).take(res_length);
        Poly {
            coeffs: coeffs.collect(),
        }
    }
}

/// Implementation of polynomial and real number division
impl<T: Clone + Div<Output = T> + PartialEq + Zero> Div<T> for Poly<T> {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        for c in &mut self.coeffs {
            *c = c.clone() / rhs.clone();
        }
        // Division with integers may leave 0 terms.
        self.trim();
        // The polynomial cannot be empty, trim has already the postcondition.
        self
    }
}

/// Implementation of polynomial and real number division
impl<T: Clone + Div<Output = T> + PartialEq + Zero> Div<T> for &Poly<T> {
    type Output = Poly<T>;

    fn div(self, rhs: T) -> Self::Output {
        let result = self.coeffs.iter().map(|x| x.clone() / rhs.clone());
        // The polynomial cannot be empty.
        Poly::new_from_coeffs_iter(result)
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
        // The polynomial cannot be empty, poly_div_impl has already the postcondition.
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
        // The polynomial cannot be empty, poly_div_impl has already the postcondition.
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
        // The polynomial cannot be empty, poly_div_impl has already the postcondition.
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
        // The polynomial cannot be empty, poly_div_impl has already the postcondition.
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

impl<T: Clone + Div<Output = T> + PartialEq + Zero> Poly<T> {
    /// In place division with a scalar
    ///
    /// # Arguments
    ///
    /// * `d` - Scalar divisor
    ///
    /// # Example
    /// ```
    /// use automatica::poly;
    /// let mut p = poly!(3, 4, 5);
    /// p.div_mut(&2);
    /// assert_eq!(poly!(1, 2, 2), p);
    /// ```
    pub fn div_mut(&mut self, d: &T) {
        for c in &mut self.coeffs {
            *c = c.clone() / d.clone();
        }
        self.trim();
    }
}

impl<T: Clone + Mul<Output = T> + One + PartialEq + Zero> Poly<T> {
    /// Calculate the power of a polynomial. With the exponentiation by squaring.
    ///
    /// #Arguments
    ///
    /// * `exp` - Positive integer exponent
    ///
    /// # Example
    /// ```
    /// use automatica::poly;
    /// let p = poly!(0, 0, 1);
    /// let pow = p.powi(4);
    /// assert_eq!(poly!(0, 0, 0, 0, 0, 0, 0, 0, 1), pow);
    /// ```
    #[must_use]
    pub fn powi(&self, exp: u32) -> Self {
        let mut n = exp;
        let mut y = Self::one();
        let mut z = (*self).clone();
        while n > 0 {
            if n % 2 == 1 {
                y = &y * &z;
            }
            z = &z * &z;
            n /= 2;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly;

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
    fn poly_add_ref() {
        let p1: Poly<i32>;
        {
            let p2 = poly!(1, 2);
            p1 = poly!(1) + &p2;
        }
        let p3 = p1 + &poly!(0, 0, 2);
        assert_eq!(poly!(2, 2, 2), p3);
        let p4 = p3 + &poly!(3);
        assert_eq!(poly!(5, 2, 2), p4);
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
    #[allow(clippy::identity_op)]
    fn poly_mul_real_number_value() {
        assert_eq!(poly!(4, 4, 3), 1 * &poly!(4, 4, 3));
        assert_eq!(poly!(10, 8, 6), &poly!(5, 4, 3) * 2);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn poly_mul_real_number_ref() {
        assert_eq!(poly!(0), poly!(4, 4, 3) * &0);
        assert_eq!(poly!(10, 8, 6), poly!(5, 4, 3) * &2);
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
        p.div_mut(&2);
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
    fn poly_pow() {
        let p = poly!(0, 0, 1);
        let pow = p.powi(4);
        assert_eq!(poly!(0, 0, 0, 0, 0, 0, 0, 0, 1), pow);
        let p2 = poly!(1, 1);
        let pow2 = p2.powi(5);
        assert_eq!(poly!(1, 5, 10, 10, 5, 1), pow2);
    }
}
