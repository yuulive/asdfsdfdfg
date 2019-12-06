extern crate automatica;

use automatica::{
    plots::bode::BodePlot,
    polynomial::Poly,
    units::{Decibel, RadiansPerSecond},
    Eval, Tf,
};

use num_complex::Complex;
use num_traits::One;

fn main() {
    let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));

    println!("T:\n{}", tf);

    let c = tf.eval(&Complex::new(0., 1.));
    println!("{}\n{}dB, {}°", c, c.norm().to_db(), c.arg().to_degrees());

    let b = tf.bode(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    for g in b.into_db_deg() {
        println!(
            "f: {:.3} rad/s, m: {:.3} dB, f: {:.1} °",
            g.angular_frequency(),
            g.magnitude(),
            g.phase()
        );
    }
}
