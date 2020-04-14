extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{poly, Tfz};

#[test]
fn left_moving_average() {
    let tf = Tfz::new(poly!(1., 1., 1.), poly!(0., 0., 3.));
    let values = [9., 8., 9., 12., 9., 12., 11., 7., 13., 9., 11., 10.];
    let arma = tf.arma_from_iter(values.iter().cloned());
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
    let arma = tf.arma_from_iter(values.iter().cloned());
    let expected = [6.4, 6.6, 6.2, 5.8, 5.6, 5.4];

    for (a, &e) in arma.skip(4).zip(&expected) {
        assert_relative_eq!(e, a);
    }
}
