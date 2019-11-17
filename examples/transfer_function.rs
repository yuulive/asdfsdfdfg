#[macro_use]
extern crate automatica;

use automatica::transfer_function::Tf;
use automatica::Eval;

use num_complex::Complex64;

fn main() {
    let tf = Tf::new(poly!(-0.75, 0.25), poly!(0.75, 0.75, 1.));

    println!("T:\n{}", tf);

    let c = tf.eval(&Complex64::new(0., 0.9));
    println!(
        "{}\n{}dB, {}°",
        c,
        20. * c.norm().log(10.),
        c.arg().to_degrees()
    );
}
