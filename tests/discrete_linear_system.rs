#[macro_use]
extern crate approx;

use automatica::{poly, signals::discrete, Poly, Ssd, Tfz};

/// TC4.1
#[test]
fn stability() {
    let stable_poles = [-0.3, 0., -0.99];
    let den = Poly::new_from_roots(&stable_poles);
    assert!(Tfz::new(poly!(0.5, 1.5), den).is_stable());

    let unstable_poles = [-1., 0.3, -5.];
    let den2 = Poly::new_from_roots(&unstable_poles);
    assert!(!Tfz::new(poly!(0.5, 1.5), den2).is_stable());
}

/// TC4.2
#[test]
fn equilibrium() {
    // Exercise 8.6
    let a = [0.6_f32, 0., 0., 0.4];
    let b = [1., 5.];
    let c = [1., 3.];
    let d = [0.];
    let sys = Ssd::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    let eq = sys.equilibrium(&[1.]).unwrap();
    assert_relative_eq!(2.5, eq.x()[0]);
    assert_relative_eq!(8.333_333, eq.x()[1]);
    assert_relative_eq!(27.5, eq.y()[0]);
}

/// TC4.3
#[test]
fn no_equilibrium() {
    // Exercise 8.6
    let a = [0.6_f32, 0., 0., 1.];
    let b = [1., 5.];
    let c = [1., 3.];
    let d = [0.];
    let sys = Ssd::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    let no_eq = sys.equilibrium(&[1.]);
    assert!(no_eq.is_none());
}

/// TC4.4
#[test]
fn static_gain() {
    // Es. 9.1
    let tf = Tfz::new(poly!(-0.5), poly!(-0.5, 1.) * poly!(-0.5, 1.));
    let sys = Ssd::new_observability_realization(&tf).unwrap();

    let step = discrete::step_vec(1., 1, 1);
    let evo = sys.evolution_fn(30, step, &[0., 0.]);
    let last = evo.last().unwrap();

    let gain = tf.static_gain();
    assert_abs_diff_eq!(gain, last.output()[0], epsilon = 1e-5);
}

/// TC4.5
#[test]
fn fir_impulse() {
    // Example 9.3
    use std::iter::{once, repeat, Iterator};

    let num = 1.016_f32 * poly!(0.015, 0.031, 0.063, 0.125, 0.25, 0.5);
    let den = poly!(0., 0., 0., 0., 0., 0., 1.);
    let g = Tfz::new(num, den);

    let mut iter = g.arma_iter(once(1.).chain(repeat(0.)));
    assert_abs_diff_eq!(0.000, iter.next().unwrap(), epsilon = 1e-3); // Step 0
    assert_abs_diff_eq!(0.508, iter.next().unwrap(), epsilon = 1e-3); // Step 1
    assert_abs_diff_eq!(0.254, iter.next().unwrap(), epsilon = 1e-3); // Step 2
    assert_abs_diff_eq!(0.127, iter.next().unwrap(), epsilon = 1e-3); // Step 3
    assert_abs_diff_eq!(0.064, iter.next().unwrap(), epsilon = 1e-3); // Step 4
    assert_abs_diff_eq!(0.031, iter.next().unwrap(), epsilon = 1e-3); // Step 5
    assert_abs_diff_eq!(0.015, iter.next().unwrap(), epsilon = 1e-3); // Step 6
    assert_abs_diff_eq!(0.000, iter.next().unwrap(), epsilon = 1e-3); // Step 7
}

/// TC4.6
#[test]
fn fir_step() {
    use std::iter::{repeat, Iterator};

    let num = 1.016_f32 * poly!(0.015, 0.031, 0.063, 0.125, 0.25, 0.5);
    let den = poly!(0., 0., 0., 0., 0., 0., 1.);
    let g = Tfz::new(num, den);

    let mut iter = g.arma_iter(repeat(1.));
    assert_abs_diff_eq!(0.000, iter.next().unwrap(), epsilon = 1e-3); // Step 0
    assert_abs_diff_eq!(0.508, iter.next().unwrap(), epsilon = 1e-3); // Step 1
    assert_abs_diff_eq!(0.762, iter.next().unwrap(), epsilon = 1e-3); // Step 2
    assert_abs_diff_eq!(0.889, iter.next().unwrap(), epsilon = 1e-3); // Step 3
    assert_abs_diff_eq!(0.953, iter.next().unwrap(), epsilon = 1e-3); // Step 4
    assert_abs_diff_eq!(0.985, iter.next().unwrap(), epsilon = 1e-3); // Step 5
    assert_abs_diff_eq!(1.000, iter.next().unwrap(), epsilon = 1e-3); // Step 6
    assert_abs_diff_eq!(1.000, iter.next().unwrap(), epsilon = 1e-3); // Step 7
}
