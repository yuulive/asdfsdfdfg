//! # Units of measurement
//!
//! List of strongly typed units of measurement. It avoids the use of primitive
//! types (newtype pattern)
//! * decibel
//! * seconds
//! * Hertz
//! * radians per second
//!
//! Conversion between units are available.

use std::{
    convert::From,
    fmt::{Display, Formatter, LowerExp, UpperExp},
};

use num_traits::{Float, FloatConst, Inv, Num};

/// Macro to implement Display trait for units. It passes the formatter options
/// to the unit inner type.
///
/// # Examples
/// ```
/// impl_display!(Seconds);
/// ```
macro_rules! impl_display {
    ($name:ident) => {
        /// Format the unit as its inner type.
        impl<T: Display + Num> Display for $name<T> {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                Display::fmt(&self.0, f)
            }
        }

        /// Format the unit as its inner type.
        impl<T: LowerExp + Float> LowerExp for $name<T> {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                LowerExp::fmt(&self.0, f)
            }
        }

        /// Format the unit as its inner type.
        impl<T: UpperExp + Float> UpperExp for $name<T> {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                UpperExp::fmt(&self.0, f)
            }
        }
    };
}

/// Trait for the conversion to decibels.
pub trait Decibel<T> {
    /// Convert to decibels
    fn to_db(&self) -> T;
}

/// Implementation of the Decibels for f64
impl Decibel<f64> for f64 {
    /// Convert f64 to decibels
    fn to_db(&self) -> Self {
        20. * self.log10()
    }
}

/// Implementation of the Decibels for f32
impl Decibel<f32> for f32 {
    /// Convert f32 to decibels
    fn to_db(&self) -> Self {
        20. * self.log10()
    }
}

/// Unit of measurement: seconds [s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Seconds<T: Num>(pub T);

/// Unit of measurement: Hertz [Hz]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hertz<T: Num>(pub T);

/// Unit of measurement: Radians per seconds [rad/s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RadiansPerSecond<T: Num>(pub T);

impl_display!(Seconds);
impl_display!(Hertz);
impl_display!(RadiansPerSecond);

impl<T: Num + FloatConst> From<Hertz<T>> for RadiansPerSecond<T> {
    /// Convert Hertz into radians per second.
    fn from(hz: Hertz<T>) -> Self {
        Self((T::PI() + T::PI()) * hz.0)
    }
}

impl<T: Num + FloatConst> From<RadiansPerSecond<T>> for Hertz<T> {
    /// Convert radians per second into Hertz.
    fn from(rps: RadiansPerSecond<T>) -> Self {
        Self(rps.0 / (T::PI() + T::PI()))
    }
}

impl<T: Inv<Output = T> + Num> Inv for Seconds<T> {
    type Output = Hertz<T>;

    /// Convert seconds into Hertz.
    fn inv(self) -> Self::Output {
        Hertz(self.0.inv())
    }
}

impl<T: Inv<Output = T> + Num> Inv for Hertz<T> {
    type Output = Seconds<T>;

    /// Convert Hertz into seconds.
    fn inv(self) -> Self::Output {
        Seconds(self.0.inv())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decibel() {
        assert_abs_diff_eq!(40., 100_f64.to_db(), epsilon = 0.);
        assert_relative_eq!(-3.0103, 2_f64.inv().sqrt().to_db(), max_relative = 1e5);

        assert_abs_diff_eq!(0., 1_f32.to_db(), epsilon = 0.);
    }

    #[test]
    fn conversion() {
        let tau = 2. * std::f64::consts::PI;
        assert_eq!(RadiansPerSecond(tau), RadiansPerSecond::from(Hertz(1.0)));

        let hz = Hertz(2.0);
        assert_eq!(hz, Hertz::from(RadiansPerSecond::from(hz)));

        let rps = RadiansPerSecond(2.0);
        assert_eq!(rps, RadiansPerSecond::from(Hertz::from(rps)));
    }

    #[quickcheck]
    fn qc_conversion_hertz(hz: f64) -> bool {
        relative_eq!(
            hz,
            Hertz::from(RadiansPerSecond::from(Hertz(hz))).0,
            max_relative = 1e-15
        )
    }

    #[quickcheck]
    fn qc_conversion_rps(rps: f64) -> bool {
        relative_eq!(
            rps,
            RadiansPerSecond::from(Hertz::from(RadiansPerSecond(rps))).0,
            max_relative = 1e-15
        )
    }

    #[quickcheck]
    fn qc_conversion_s_hz(s: f32) -> bool {
        // f32 precision.
        relative_eq!(s, Seconds(s).inv().inv().0, max_relative = 1e-5)
    }

    #[quickcheck]
    fn qc_conversion_hz_s(hz: f64) -> bool {
        // f64 precision.
        relative_eq!(hz, Hertz(hz).inv().inv().0, max_relative = 1e-14)
    }

    #[test]
    fn format() {
        assert_eq!("0.33".to_owned(), format!("{:.2}", Seconds(1. / 3.)));
        assert_eq!("0.3333".to_owned(), format!("{:.*}", 4, Seconds(1. / 3.)));
        assert_eq!("4.20e1".to_owned(), format!("{:.2e}", Seconds(42.)));
        assert_eq!("4.20E2".to_owned(), format!("{:.2E}", Seconds(420.)));
    }
}
