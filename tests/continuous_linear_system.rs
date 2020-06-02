extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{poly, signals::continuous, Poly, Seconds, Ss, Tf};

#[test]
fn stability() {
    let stable_poles = [-1., -1.3, -15.];
    let den = Poly::new_from_roots(&stable_poles);
    assert!(Tf::new(poly!(0.5, 1.5), den).is_stable());

    let unstable_poles = [-1., 0.3, -5.];
    let den2 = Poly::new_from_roots(&unstable_poles);
    assert!(!Tf::new(poly!(0.5, 1.5), den2).is_stable());
}

#[test]
fn equilibrium() {
    // Es 3.7
    let a = [0., 1., -1., -1.];
    let b = [0., 1.];
    let c = [0., 1.];
    let d = [0.];
    let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    let eq = sys.equilibrium(&[1.]).unwrap();
    assert_eq!(&[1., 0.], eq.x());
    assert_eq!(&[0.], eq.y());

    // Es 3.9
    let a = [0., 0., 1., 2.];
    let b = [1., 2., 3., 4.];
    let c = [5., 6.];
    let d = [0., 0.];
    let sys = Ss::new_from_slice(2, 2, 1, &a, &b, &c, &d);
    let no_eq = sys.equilibrium(&[1., 1.]);
    assert!(no_eq.is_none());
}

#[test]
fn static_gain() {
    // 5.4.4
    let tf = Tf::new(poly!(1., 5.), poly!(1., 2.) * poly!(1., 1.));
    let sys = Ss::new_observability_realization(&tf).unwrap();

    let step = continuous::step(1., 1);
    let evo = sys.rk2(&step, &[0., 0.], Seconds(0.1), 150);
    let last = evo.last().unwrap();

    let gain = tf.static_gain();
    assert_abs_diff_eq!(gain, last.output()[0], epsilon = 1e-2);
}
