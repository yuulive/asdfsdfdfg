#[macro_use]
extern crate automatica;

use automatica::{units::ToDecibel, Tf};

use num_complex::Complex64;

#[allow(clippy::non_ascii_literal)]
fn main() {
    let tf = Tf::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));

    println!("T:\n{}", tf);

    let c = tf.eval(&Complex64::new(0., 0.9));
    println!("\nEvaluation at s = 0 + 0.9i:");
    println!(
        "{:.3} = {:.3}dB, {:.3}°",
        c,
        c.norm().to_db(),
        c.arg().to_degrees()
    );

    println!("\nStatic Gain: {:.3}", tf.static_gain());
}
