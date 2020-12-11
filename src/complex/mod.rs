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
}
