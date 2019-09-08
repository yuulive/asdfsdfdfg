//! # Units of measurement
//!
//! List of strongly typed units of measurements. It avoids the use of primitive
//! types.

use std::convert::From;

/// 2π
const TAU: f64 = 2. * std::f64::consts::PI;

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
        assert_eq!(40., 100_f64.to_db());
        assert_eq!(-3.0102999566398116, 2_f64.inv().sqrt().to_db());
    }

    #[test]
    fn conversion() {
        assert_eq!(RadiantsPerSecond(TAU), RadiantsPerSecond::from(Hertz(1.0)));

        let hz = Hertz(2.0);
        assert_eq!(hz, Hertz::from(RadiantsPerSecond::from(hz)));

        let rps = RadiantsPerSecond(2.0);
        assert_eq!(rps, RadiantsPerSecond::from(Hertz::from(rps)));
    }
}
