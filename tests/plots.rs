extern crate automatica;
#[macro_use]
extern crate approx;

use automatica::{
    plots::{bode::BodePlot, polar::PolarPlot},
    poly, Poly, RadiansPerSecond, Tf,
};

#[test]
fn bode_plot() {
    let xi = 0.1_f32;
    let omega = 1.;
    let tf = Tf::new(
        poly!(1.),
        poly!(1., 2. * xi / omega, (omega * omega).recip()),
    );

    let bode = tf.bode(RadiansPerSecond(0.1), RadiansPerSecond(10.), 0.1);
    let data: Vec<_> = bode.into_db_deg().collect();
    println!("{:?}", data[10]);

    // at resonance frequency
    assert!(data[10].magnitude() > 10.);
    assert_relative_eq!(-90., data[10].phase());
}

#[test]
fn polar_plot() {
    let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));

    println!("T:\n{}\n", tf);

    let p = tf.polar(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);

    assert!(p.clone().all(|x| x.magnitude() < 1.));
    // Assert that the values are decreasing.
    assert!(
        p.fold((true, 1.0), |acc, x| (
            acc.0 && x.magnitude() < acc.1,
            x.magnitude()
        ))
        .0
    );
}

#[test]
fn root_locus_plot() {
    // Example 13.2.
    let tf = Tf::new(poly!(1.0_f64), Poly::new_from_roots(&[0., -3., -5.]));
    // println!("{}", &tf);

    let loci = tf.root_locus_iter(1., 130., 1.);
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
