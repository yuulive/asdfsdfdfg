extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{poly, Eval, Poly};

#[test]
fn test_name() {
    let p = poly!(1., 2., 3.);
    assert_eq!(1., p.eval(0.));
}

#[test]
fn roots_consistency() {
    // Wilkinson's polynomial.
    let wp = Poly::new_from_roots(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    ]);

    // Roots with Aberth-Ehrlich Method.
    let mut roots = wp.iterative_roots();
    roots.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());

    // Roots with eigenvalue decomposition.
    let mut roots2 = wp.complex_roots();
    roots2.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());

    for (i, e) in roots.iter().zip(roots2) {
        assert_relative_eq!(i.re, e.re, max_relative = 1e-2);
        assert_relative_eq!(i.im, e.im);
    }
}
