extern crate automatica;

use automatica::plots::bode::BodePlot;
use automatica::transfer_function::Tf;
use automatica::{polynomial::Poly, Decibel, Eval};

use num_complex::Complex;

fn main() {
    let tf = Tf::new(Poly::new_from_coeffs(&[1.]), Poly::new_from_roots(&[-1.]));

    println!("T:\n{}", tf);

    let c = tf.eval(&Complex::new(0., 1.));
    println!("{}\n{}dB, {}°", c, c.norm().to_db(), c.arg().to_degrees());

    let b = tf.bode(0.1, 10.0, 0.1);
    for g in b.into_db_deg() {
        println!(
            "f: {:.3} rad, m: {:.3} dB, f: {:.1} °",
            g.angular_frequency(),
            g.magnitude(),
            g.phase()
        );
    }
}