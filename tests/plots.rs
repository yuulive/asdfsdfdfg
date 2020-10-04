extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{
    plots::{bode::BodeT, polar::PolarT},
    poly, Poly, RadiansPerSecond, Tf,
};

/// TC5.1
#[test]
fn bode_plot() {
    // Figure 7.8
    use crate::automatica::units::ToDecibel;

    let xi = 0.1_f32;
    let omega = 1.;
    let tf = Tf::new(
        poly!(1.),
        poly!(1., 2. * xi / omega, (omega * omega).recip()),
    );

    let bode = tf.bode(RadiansPerSecond(0.1), RadiansPerSecond(10.), 0.1);
    let data: Vec<_> = bode.into_iter().into_db_deg().collect();

    // At resonance frequency, 1 rad/s is the 10th element of the iterator.
    let peak = (1. / 2. / xi.abs()).to_db();
    assert_relative_eq!(peak, data[10].magnitude(), max_relative = 1e-6);
    assert_relative_eq!(-90., data[10].phase());
}

/// TC5.2
#[test]
fn polar_plot() {
    let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));
    let p = tf.polar(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    let points = p.into_iter();

    assert!(points.clone().all(|x| x.magnitude() < 1.));
    // Assert that the values are decreasing.
    assert!(
        points
            .fold((true, 1.0), |acc, x| (
                acc.0 && x.magnitude() < acc.1,
                x.magnitude()
            ))
            .0
    );
}

/// TC5.3
#[test]
fn root_locus_plot() {
    // Example 13.2.
    let tf = Tf::new(poly!(1.0_f64), Poly::new_from_roots(&[0., -3., -5.]));

    let loci = tf.root_locus_plot(1., 130., 1.);
    for locus in loci {
        let out = locus.output();
        if locus.k() < 120. {
            assert!(out[0].re < 0.);
            assert!(out[1].re < 0.);
            assert!(out[2].re < 0.);
        } else {
            assert!(out[0].re > 0.);
            assert!(out[1].re > 0.);
            assert!(relative_eq!(out[2].re, -8.) || out[2].re <= -8.);
        }
        // Test symmetry
        assert_relative_eq!(out[0].im.abs(), out[1].im.abs());
        assert_relative_eq!(out[2].im, 0.);
    }
}
