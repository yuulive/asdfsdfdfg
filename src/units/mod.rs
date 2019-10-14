//! # Units of measurement
//!
//! List of strongly typed units of measurement. It avoids the use of primitive
//! types.

use std::convert::From;

/// 2π
const TAU: f64 = 2. * std::f64::consts::PI;

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
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }

        /// Format the unit as its inner type.
        impl std::fmt::LowerExp for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }

        /// Format the unit as its inner type.
        impl std::fmt::UpperExp for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self.0.fmt(f)
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

/// Unit of measurement: seconds [s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Seconds(pub f64);

/// Unit of measurement: Hertz [Hz]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Hertz(pub f64);

/// Unit of measurement: Radiants per seconds [rad/s]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct RadiantsPerSecond(pub f64);

impl_display!(Seconds);
impl_display!(Hertz);
impl_display!(RadiantsPerSecond);

impl From<Hertz> for RadiantsPerSecond {
    fn from(hz: Hertz) -> Self {
        Self(TAU * hz.0)
    }
}

impl From<RadiantsPerSecond> for Hertz {
    fn from(rps: RadiantsPerSecond) -> Self {
        Self(rps.0 / TAU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ops::inv::Inv;

    #[test]
    fn decibel() {
        assert_abs_diff_eq!(40., 100_f64.to_db(), epsilon = 0.);
        assert_relative_eq!(-3.0103, 2_f64.inv().sqrt().to_db(), max_relative = 1e5);
    }

    #[test]
    fn conversion() {
        assert_eq!(RadiantsPerSecond(TAU), RadiantsPerSecond::from(Hertz(1.0)));

        let hz = Hertz(2.0);
        assert_eq!(hz, Hertz::from(RadiantsPerSecond::from(hz)));

        let rps = RadiantsPerSecond(2.0);
        assert_eq!(rps, RadiantsPerSecond::from(Hertz::from(rps)));
    }

    #[test]
    fn format() {
        assert_eq!("0.33".to_owned(), format!("{:.2}", Seconds(1. / 3.)));
        assert_eq!("0.3333".to_owned(), format!("{:.*}", 4, Seconds(1. / 3.)));
        assert_eq!("4.20e1".to_owned(), format!("{:.2e}", Seconds(42.)));
        assert_eq!("4.20E2".to_owned(), format!("{:.2E}", Seconds(420.)));
    }
}
