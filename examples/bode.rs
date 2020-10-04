extern crate automatica;

use num_complex::Complex;
use num_traits::One;

use automatica::{
    plots::bode::BodeT,
    units::{RadiansPerSecond, ToDecibel},
    Poly, Tf,
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
    let b = tf.bode(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    for g in b.into_iter().into_db_deg() {
        println!(
            "f: {:.3} rad/s, m: {:.3} dB, f: {:.1} °",
            g.angular_frequency(),
            g.magnitude(),
            g.phase()
        );
    }
}
