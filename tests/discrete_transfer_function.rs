#[macro_use]
extern crate approx;

use automatica::{poly, Tfz};

/// TC4.7
#[test]
fn left_moving_average() {
    let tf = Tfz::new(poly!(1., 1., 1.), poly!(0., 0., 3.));
    let values = [9., 8., 9., 12., 9., 12., 11., 7., 13., 9., 11., 10.];
    let arma = tf.arma_iter(values.iter().cloned());
    let expected = [8.667, 9.667, 10., 11., 10.667, 10., 10.333, 9.667, 11., 10.];

    for (a, &e) in arma.skip(2).zip(&expected) {
        dbg!(a, e);
        assert_abs_diff_eq!(e, a, epsilon = 0.001);
    }
}

#[test]
fn central_moving_average() {
    let tf = Tfz::new(poly!(1., 1., 1., 1., 1.), poly!(0., 0., 5.));
    let values = [4., 6., 5., 8., 9., 5., 4., 3., 7., 8.];
    let arma = tf.arma_iter(values.iter().cloned());
    let expected = [6.4, 6.6, 6.2, 5.8, 5.6, 5.4];

    for (a, &e) in arma.skip(4).zip(&expected) {
        assert_relative_eq!(e, a);
    }
}

#[test]
fn arma_channel_example() {
    let tf = Tfz::new(poly!(1.), poly!(1., 0.5));
    let values = &[0.1, 0.3, 0.6, 0.8, 1.0];
    let arma = tf.arma_iter(values.iter().cloned());
    let expected = [0., 0.2, 0.2, 0.8, 0.];

    for (a, &e) in arma.skip(4).zip(&expected) {
        assert_relative_eq!(e, a);
    }
}
