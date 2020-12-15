//! Module that defines extension for complex numbers.

use num_complex::Complex;
use num_traits::Float;

/// Calculate the natural pulse of a complex number, it corresponds to its modulus.
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

/// Calculate the damping of a complex number, it corresponds to the cosine of the
/// angle between the segment joining the complex number to the origin and the
/// real negative semiaxis.
///
/// By definition the damping of 0+0i is -1.
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
        // Handle the case where the pulse is zero to avoid division by zero.
        -T::one()
    } else {
        -c.re / w
    }
}

/// Division between complex numbers that avoids overflows.
///
/// # Arguments
///
/// * `a` - Dividend
/// * `b` - Divisor
/// Michael Baudin, Robert L. Smith, A Robust Complex Division in Scilab, 2012, arXiv:1210.4539v2 [cs.MS]
pub(crate) fn compdiv<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if b.im.abs() <= b.re.abs() {
        let (e, f) = compdiv_impl(a.re, a.im, b.re, b.im);
        Complex::new(e, f)
    } else {
        // Real and imaginary parts shall be swapped.
        let (e, f) = compdiv_impl(a.im, a.re, b.im, b.re);
        // And the imaginary part shall change sign.
        Complex::new(e, -f)
    }
}

/// Implementation of division between complex numbers.
///
/// # Arguments
///
/// * `a` - Dividend real part if Im{divisor} <= Re{divisor} else imaginary
/// * `b` - Dividend imaginary part if Im{divisor} <= Re{divisor} else real
/// * `c` - Divisor real part if Im{divisor} <= Re{divisor} else imaginary
/// * `d` - Divisor imaginary part if Im{divisor} <= Re{divisor} else real
/// Michael Baudin, Robert L. Smith, A Robust Complex Division in Scilab, 2012, arXiv:1210.4539v2 [cs.MS]
#[allow(clippy::many_single_char_names)]
fn compdiv_impl<T: Float>(a: T, b: T, c: T, d: T) -> (T, T) {
    let r = d / c;
    let t = (c + d * r).recip();
    if r.is_zero() {
        let e = (a + d * (b / c)) * t;
        let f = (b - d * (a / c)) * t;
        (e, f)
    } else {
        let e = (a + b * r) * t;
        let f = (b - a * r) * t;
        (e, f)
    }
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

    fn p2(n: i32) -> f64 {
        2.0_f64.powf(f64::from(n))
    }

    #[test]
    fn complex_division_a() {
        let a = compdiv(Complex::new(1., 1.), Complex::new(1., 1e307));
        assert_eq!(
            Complex::new(1.000_000_000_000_000_1e-307, -1.000_000_000_000_000_1e-307),
            a
        );
    }

    #[test]
    fn complex_division_b() {
        let b = compdiv(Complex::new(1., 1.), Complex::new(1e-307, 1e-307));
        assert_eq!(Complex::new(1.000_000_000_000_000_1e307, 0.), b);
    }

    #[test]
    fn complex_division_c() {
        let c = compdiv(Complex::new(1e307, 1e-307), Complex::new(1e204, 1e-204));
        assert_eq!(Complex::new(1e103, -1e-305), c);
    }

    #[test]
    fn complex_division_1() {
        let d1 = compdiv(Complex::new(1., 1.), Complex::new(1., p2(1023)));
        assert_eq!(Complex::new(p2(-1023), -p2(-1023)), d1);
    }

    #[test]
    fn complex_division_2() {
        let d2 = compdiv(Complex::new(1., 1.), Complex::new(p2(-1023), p2(-1023)));
        assert_eq!(Complex::new(p2(1023), 0.), d2);
    }

    #[test]
    fn complex_division_3() {
        let d3 = compdiv(
            Complex::new(p2(1023), p2(-1023)),
            Complex::new(p2(677), p2(-677)),
        );
        assert_eq!(Complex::new(p2(346), -p2(-1008)), d3);
    }

    #[test]
    fn complex_division_5() {
        let d5 = compdiv(
            Complex::new(p2(1020), p2(-844)),
            Complex::new(p2(656), p2(-780)),
        );
        assert_eq!(Complex::new(p2(364), -p2(-1072)), d5);
    }

    #[test]
    fn complex_division_6() {
        let d6 = compdiv(
            Complex::new(p2(-71), p2(1021)),
            Complex::new(p2(1001), p2(-323)),
        );
        assert_eq!(Complex::new(p2(-1072), p2(20)), d6);
    }

    #[test]
    fn complex_division_limits() {
        let c1 = Complex::new(1., 1.);
        let c2 = Complex::new(1., -1.);
        let c3 = Complex::new(-1., 1.);
        let c4 = Complex::new(-1., -1.);

        let zero = Complex::new(0., 0.);

        assert!(compdiv(c1, zero).is_nan());
        assert!(compdiv(c2, zero).is_nan());
        assert!(compdiv(c3, zero).is_nan());
        assert!(compdiv(c4, zero).is_nan());
    }
}
