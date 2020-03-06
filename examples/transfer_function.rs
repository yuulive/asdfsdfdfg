#[macro_use]
extern crate automatica;

use automatica::{Eval, Tf};

use num_complex::Complex64;

#[allow(clippy::non_ascii_literal)]
fn main() {
    let tf = Tf::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));

    println!("T:\n{}", tf);

    let c = tf.eval(Complex64::new(0., 0.9));
    println!("\nEvaluation at s = 0 + 0.9i:");
    println!(
        "{}\n{}dB, {}°",
        c,
        20. * c.norm().log(10.),
        c.arg().to_degrees()
    );

    println!("\nStatic Gain: {:.3}", tf.static_gain());
}
