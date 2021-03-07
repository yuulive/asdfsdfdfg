#[macro_use]
extern crate approx;

use automatica::{
    num_complex::Complex,
    num_traits::{One, Zero},
    poly, Poly,
};

/// TC1.1
#[test]
fn multiplicative_unity() {
    let p1 = poly!(1., 0., 0.3, -4.);
    let one_p = poly!(1.);
    // Polynomial unity.
    assert_eq!(p1, &p1 * &one_p);

    let p2 = poly!(1., 0., 0.3, -4.);
    // Scalar unity.
    assert_eq!(p2, &p2 * 1.);

    let zero_p = poly!(0.);
    assert_eq!(zero_p, &zero_p * 1. * Poly::one());
}

/// TC1.2
#[test]
fn multiplicative_null() {
    let p1 = poly!(1., 0., 0.3, -4.);
    let zero_p = poly!(0.);
    // Polynomial zero.
    assert_eq!(zero_p, &p1 * &zero_p);

    let p2 = poly!(1., 0., 0.3, -4.);
    // Scalar zero.
    assert_eq!(zero_p, &p2 * 0.);

    assert_eq!(zero_p, &zero_p * 0. * Poly::zero());
}

/// TC1.4
#[test]
fn additive_invariant() {
    let p1 = poly!(0., -4.5, 0.6);
    let zero_p = poly!(0.);
    // Polynomial zero.
    assert_eq!(p1, &p1 + &zero_p);
    // Scalar zero.
    assert_eq!(p1, &p1 - 0.);
}

/// TC1.6
#[test]
fn roots_degree() {
    let p0 = poly!(1);
    assert_eq!(Some(0), p0.degree());
    let p1 = poly!(1, 2);
    assert_eq!(Some(1), p1.degree());
    let p2 = poly!(1, 2, 3);
    assert_eq!(Some(2), p2.degree());
}

/// TC1.7
#[test]
fn no_degree() {
    let p0 = poly!(0);
    assert_eq!(None, p0.degree());
}

/// TC1.5
#[test]
fn additive_inverse() {
    let p1 = poly!(0, -4, 6);
    let p2 = poly!(1, 44, -12);
    let sum = &p1 + &p2;
    assert_eq!(p1, sum - p2);
}

/// TC1.3
#[test]
fn multiplicative_inverse() {
    let p1 = poly!(0., -4., 6.);
    let p2 = poly!(1., 44., -12.);
    let mul = &p1 * &p2;
    assert_eq!(p1, mul / p2);
}

/// TC1.8
#[test]
fn derivation() {
    let p2 = poly!(0., 2., 3.);
    let p1 = p2.derive();
    assert_eq!(Some(1), p1.degree());

    let p0 = p1.derive();
    assert_eq!(Some(0), p0.degree());

    let p0_der = p0.derive();
    assert_eq!(None, p0_der.degree());
}

/// TC1.9
#[test]
fn integration() {
    let p0 = poly!(0.);
    let p1 = p0.integrate(1.);
    assert_eq!(Some(0), p1.degree());

    let p2 = p1.integrate(-1.);
    assert_eq!(Some(1), p2.degree());

    let p3 = p2.integrate(2.);
    assert_eq!(Some(2), p3.degree());
}

#[test]
fn arithmetics() {
    let p1 = poly!(1, 1, 1);
    let p2 = poly!(-1, -1, -1);
    let result = p1 + p2;
    assert_eq!(Poly::<i32>::zero(), result);

    let p3 = poly!(1., 1., 1., 1., 1.);
    let p4 = poly!(-1., 0., 1.);
    let quotient = &p3 / &p4;
    let reminder = &p3 % &p4;
    assert_eq!(poly!(2., 1., 1.), quotient);
    assert_eq!(poly!(3., 2.), reminder);

    let original = p4.mul_fft(quotient) + reminder;
    assert_eq!(p3.degree(), original.degree());
    for i in 0..=original.degree().unwrap() {
        assert_relative_eq!(p3[i], original[i]);
    }
}

/// TC1.10
#[test]
fn maximum_minimum() {
    let cubic = Poly::<f32>::new_from_roots(&[-1., 0., 1.]);
    let slope = cubic.derive();
    let mut stationary = slope.real_roots().unwrap();
    stationary.sort_by(|x, y| x.partial_cmp(y).unwrap());

    // Test roots of derivative.
    assert_relative_eq!(-0.57735, stationary[0], max_relative = 1e-5);
    assert_relative_eq!(0.57735, stationary[1], max_relative = 1e-5);

    let curvature = slope.derive();

    // Local maximum.
    assert!(curvature.eval(&stationary[0]).is_sign_negative());
    // Local minimum.
    assert!(curvature.eval(&stationary[1]).is_sign_positive());
}

#[test]
fn roots_consistency() {
    // Wilkinson's polynomial.
    let roots = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    ];
    let wp = Poly::new_from_roots(&roots);

    // Roots with Aberth-Ehrlich Method.
    let mut iter_roots = wp.iterative_roots();
    iter_roots.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());
    for (i, r) in iter_roots.iter().zip(&roots) {
        assert_relative_eq!(i.re, *r, max_relative = 1e-3);
        assert_relative_eq!(i.im, 0.);
    }

    // Roots with eigenvalue decomposition.
    let mut eig_roots = wp.complex_roots();
    eig_roots.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());
    for (i, r) in eig_roots.iter().zip(&roots) {
        assert_relative_eq!(i.re, *r, max_relative = 1e-3);
        assert_relative_eq!(i.im, 0.);
    }
}

#[test]
fn chebyshev_first_kind() {
    // Recurrence relation:
    // T0(x) = 1
    // T1(x) = x
    // T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
    let mut polys: Vec<Poly<i32>> = Vec::new();
    polys.push(Poly::<i32>::one());
    polys.push(poly!(0, 1));
    let c = poly!(0, 2);
    for n in 2..12 {
        let tmp = &c * &polys[n - 1];
        polys.push(&tmp - &polys[n - 2]);
    }

    let t2 = poly!(-1, 0, 2);
    assert_eq!(t2, polys[2]);

    let t3 = poly!(0, -3, 0, 4);
    assert_eq!(t3, polys[3]);

    let t4 = poly!(1, 0, -8, 0, 8);
    assert_eq!(t4, polys[4]);

    let t5 = poly!(0, 5, 0, -20, 0, 16);
    assert_eq!(t5, polys[5]);

    let t6 = poly!(-1, 0, 18, 0, -48, 0, 32);
    assert_eq!(t6, polys[6]);

    let t7 = poly!(0, -7, 0, 56, 0, -112, 0, 64);
    assert_eq!(t7, polys[7]);

    let t8 = poly!(1, 0, -32, 0, 160, 0, -256, 0, 128);
    assert_eq!(t8, polys[8]);

    let t9 = poly!(0, 9, 0, -120, 0, 432, 0, -576, 0, 256);
    assert_eq!(t9, polys[9]);

    let t10 = poly!(-1, 0, 50, 0, -400, 0, 1120, 0, -1280, 0, 512);
    assert_eq!(t10, polys[10]);

    let t11 = poly!(0, -11, 0, 220, 0, -1232, 0, 2816, 0, -2816, 0, 1024);
    assert_eq!(t11, polys[11]);
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn chebyshev_roots() {
    let polys = chebyshev_polys();
    for (n, t) in polys.iter().enumerate().take(12).skip(2) {
        let mut roots: Vec<_> = (1..=n)
            .map(|i| chebyshev_nodes(i as f64, n as f64))
            .collect();

        // Roots with Aberth-Ehrlich Method.
        let mut iter_roots = t.iterative_roots();
        iter_roots.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());
        roots.sort_by(|&x, &y| x.partial_cmp(&y).unwrap());
        for (i, r) in iter_roots.iter().zip(&roots) {
            assert_relative_eq!(i.re, *r, max_relative = 1e-10, epsilon = 1e-10);
            assert_relative_eq!(i.im, 0.);
        }
    }
}

fn chebyshev_nodes(i: f64, n: f64) -> f64 {
    ((2. * i - 1.) * std::f64::consts::PI / (2. * n)).cos()
}

fn chebyshev_polys() -> Vec<Poly<f64>> {
    // Recurrence relation:
    // T0(x) = 1
    // T1(x) = x
    // T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
    let mut polys: Vec<Poly<f64>> = Vec::new();
    polys.push(Poly::<f64>::one());
    polys.push(poly!(0., 1.));
    let c = poly!(0., 2.);
    for n in 2..12 {
        let tmp = &c * &polys[n - 1];
        polys.push(&tmp - &polys[n - 2]);
    }
    polys
}

/// TC1.11
#[test]
fn nearly_multiple_zeros() {
    let p4 = Poly::new_from_roots(&[0.1, 0.1, 0.1, 0.5, 0.6, 0.7]);
    let r4 = p4.iterative_roots();
    assert!(r4.iter().all(|c| relative_eq!(c.im, 0.)));
    let p4n = Poly::new_from_roots_iter(r4.iter().map(|r| r.re));
    for (c1, c2) in p4.as_slice().iter().zip(p4n.as_slice()) {
        assert_relative_eq!(c1, c2, max_relative = 1e-4);
    }

    let p5 = Poly::new_from_roots(&[0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4]);
    let r5 = p5.iterative_roots();
    assert!(r5.iter().all(|c| relative_eq!(c.im, 0.)));
    let p5n = Poly::new_from_roots_iter(r5.iter().map(|r| r.re));
    for (c1, c2) in p5.as_slice().iter().zip(p5n.as_slice()) {
        assert_relative_eq!(c1, c2, max_relative = 1e-1);
    }

    let p6 = Poly::new_from_roots(&[0.1, 1.001, 0.998, 1.00002, 0.99999]);
    let r6 = p6.iterative_roots();
    assert!(r6.iter().all(|c| relative_eq!(c.im, 0.)));
    let p6n = Poly::new_from_roots_iter(r6.iter().map(|r| r.re));
    for (c1, c2) in p6.as_slice().iter().zip(p6n.as_slice()) {
        assert_relative_eq!(c1, c2, max_relative = 1e-2);
    }

    let p8 = Poly::new_from_roots(&[-1., -1., -1., -1., -1.]);
    let r8 = p8.iterative_roots();
    assert!(r8.iter().all(|c| relative_eq!(c.im, 0.)));
    assert!(r8
        .iter()
        .all(|c| relative_eq!(c.re, -1., max_relative = 1e-3)));
}

/// TC1.12
#[test]
fn equimodular_zeros() {
    // Roots are equispaced on circle of radius 0.01.
    let p9_1 = Poly::new_from_coeffs(&[-1e-20, 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]);
    // Roots are equispaced on circle of radius 100.
    let p9_2 = Poly::new_from_coeffs(&[1e20, 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]);
    let p9 = p9_1 * p9_2;

    let r9 = p9.iterative_roots();
    for r in &r9 {
        assert!(relative_eq!(r.norm(), 100.) || relative_eq!(r.norm(), 0.01));
    }
    assert_eq!(20, r9.len());
}

/// TC1.13
#[allow(clippy::similar_names)]
#[test]
fn defects_in_algorithm() {
    let a_1: f64 = 1e3;
    let p10_1 = Poly::new_from_roots(&[a_1, 1., a_1.recip()]);
    let r10_1 = p10_1.iterative_roots();
    assert!(r10_1.iter().all(|c| relative_eq!(c.im, 0.)));
    assert!(r10_1.contains(&Complex::new(1e-3, 0.)));
    assert!(r10_1.contains(&Complex::new(1., 0.)));
    assert!(r10_1.contains(&Complex::new(1e3, 0.)));

    let a_2: f64 = 1e6;
    let p10_2 = Poly::new_from_roots(&[a_2, 1., a_2.recip()]);
    let r10_2 = p10_2.iterative_roots();
    assert!(r10_2.iter().all(|c| relative_eq!(c.im, 0.)));
    assert!(r10_2.contains(&Complex::new(1e-6, 0.)));
    assert!(r10_2.contains(&Complex::new(1., 0.)));
    assert!(r10_2.contains(&Complex::new(1e6, 0.)));

    let a_3: f64 = 1e9;
    let p10_3 = Poly::new_from_roots(&[a_3, 1., a_3.recip()]);
    let r10_3 = p10_3.iterative_roots();
    assert!(r10_3.iter().all(|c| relative_eq!(c.im, 0.)));
    assert!(r10_3.contains(&Complex::new(1e-9, 0.)));
    assert!(r10_3.contains(&Complex::new(1., 0.)));
    assert!(r10_3.contains(&Complex::new(1e9, 0.)));
}

/// TC1.14
#[allow(clippy::similar_names)]
#[test]
fn defects_on_circle() {
    use std::iter;
    // Roots on circle,
    // r = circle radius, n = polynomial degree
    let c = |r: f64, n: usize| {
        iter::once(-r)
            .chain(iter::repeat(0.).take(n - 1))
            .chain(iter::once(1.))
    };

    // n roots of 0.9^n are on a circle of radius 0.9
    let n_1 = 0.9_f64.powi(30);
    let p11_1 = Poly::new_from_coeffs_iter(c(1., 30)) * Poly::new_from_coeffs_iter(c(n_1, 30));
    let r11_1 = p11_1.iterative_roots();
    for r in &r11_1 {
        assert!(relative_eq!(r.norm(), 1.) || relative_eq!(r.norm(), 0.9));
    }

    let n_2 = 0.9_f64.powi(40);
    let p11_2 = Poly::new_from_coeffs_iter(c(1., 40)) * Poly::new_from_coeffs_iter(c(n_2, 40));
    let r11_2 = p11_2.iterative_roots();
    for r in &r11_2 {
        assert!(relative_eq!(r.norm(), 1.) || relative_eq!(r.norm(), 0.9));
    }

    let n_3 = 0.9_f64.powi(50);
    let p11_3 = Poly::new_from_coeffs_iter(c(1., 50)) * Poly::new_from_coeffs_iter(c(n_3, 50));
    let r11_3 = p11_3.iterative_roots();
    for r in &r11_3 {
        assert!(relative_eq!(r.norm(), 1.) || relative_eq!(r.norm(), 0.9));
    }
}
