extern crate automatica;
#[macro_use]
extern crate approx;

use std::str::FromStr;

use num_complex::Complex;

use automatica::{damp, pulse};

/// TC6.1
#[test]
fn damping_of_zero() {
    let zero = Complex::from_str("0").unwrap();
    assert_relative_eq!(0., pulse(zero));
    assert_relative_eq!(-1., damp(zero));
}
