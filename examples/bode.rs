extern crate au;

use num_complex::Complex;
use num_traits::One;

use au::{
    plots::bode::Bode,
    poly,
    units::{RadiansPerSecond, ToDecibel},
    Poly, Tf, Tfz,
};

#[allow(clippy::non_ascii_literal)]
fn main() {
    let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));

    println!("T:\n{}", tf);

    let c = tf.eval(&Complex::new(0., 1.));
    println!("\nEvaluation at i:");
    println!(
        "{} = {:.3}dB, {:.3}°",
        c,
        c.norm().to_db(),
        c.arg().to_degrees()
    );

    println!("\nBode Plot:");
    let b = Bode::new(tf, RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    for g in b.into_iter().into_db_deg() {
        println!(
            "f: {:.3} rad/s, m: {:.3} dB, f: {:.1} °",
            g.angular_frequency(),
            g.magnitude(),
            g.phase()
        );
    }

    let k = 0.5;
    let tfz = Tfz::new(poly!(1. - k), poly!(-k, 1.));
    println!("\nDiscrete function T:\n{}\n", tfz);
    let pz = Bode::new_discrete(tfz, RadiansPerSecond(0.01), 0.1);
    for g in pz.into_iter().into_db_deg() {
        println!(
            "f: {:.3} rad/s, m: {:.3} dB, f: {:.1} °",
            g.angular_frequency(),
            g.magnitude(),
            g.phase()
        );
    }
}
