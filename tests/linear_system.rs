extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{poly, signals::continuous, Poly, Seconds, Ss, Ssd, Tf, Tfz};

#[test]
fn poles_eigenvalues() {
    let num = poly!(4.);
    let den = Poly::new_from_roots(&[-1., -2., -3.]);
    let tf = Tf::new(num, den).normalize();

    let sys = Ss::new_observability_realization(&tf).unwrap();

    assert_eq!(tf.complex_poles(), sys.poles());
}

#[test]
fn series_system() {
    let tf1 = Tfz::new(poly!(1.), Poly::new_from_roots(&[-0.7, -0.5]));
    let tf2 = Tfz::new(poly!(1.), Poly::new_from_roots(&[-0.2, -0.25]));

    let stable_tf = &tf1 * &tf2;
    assert!(stable_tf.is_stable());

    let tf3 = Tfz::new(poly!(1.), Poly::new_from_roots(&[2., -0.25]));

    let unstable_tf = tf1 * tf3;
    assert!(!unstable_tf.is_stable());
}

#[test]
fn parallel_system() {
    let tf1 = Tf::new(poly!(1.), Poly::new_from_roots(&[-1., -0.5]));
    let tf2 = Tf::new(poly!(1.), Poly::new_from_roots(&[-2., -0.25]));

    let stable_tf = &tf1 + &tf2;
    assert!(stable_tf.is_stable());

    let tf3 = Tf::new(poly!(1.), Poly::new_from_roots(&[2., -0.25]));

    let unstable_tf = tf1 + tf3;
    assert!(!unstable_tf.is_stable());
}

#[test]
fn initial_state_independence() {
    let a = &[0.3_f32, 0., 0., 0.25];
    let b = &[3., -1.];
    let c = &[1., 1.];
    let d = &[1.];
    let sys = Ssd::new_from_slice(2, 1, 1, a, b, c, d);

    let tf = Tfz::<f32>::new_from_siso(&sys).unwrap();
    let expected = tf.static_gain();

    let iter = std::iter::repeat(vec![1.]);
    let steps = 30;

    let evo = sys.evolution_iter(iter.clone(), &[0., 0.]);
    let a = evo.take(steps).last().unwrap()[0];
    assert_relative_eq!(expected, a);

    let evo2 = sys.evolution_iter(iter, &[1., -1.]);
    let b = evo2.take(steps).last().unwrap()[0];
    assert_relative_eq!(expected, b);
}

#[test]
fn to_zero() {
    // 5.4.4
    let tf = Tf::new(poly!(1., 5.), poly!(1., 2.) * poly!(1., 1.));
    let sys = Ss::new_observability_realization(&tf).unwrap();

    let impulse = continuous::impulse(1., Seconds(0.), 1);
    let evo = sys.rk2(&impulse, &[0., 0.], Seconds(0.1), 150);
    let last = evo.last().unwrap();

    assert_abs_diff_eq!(0., last.state()[0], epsilon = 1e-4);
}
