extern crate automatica;

use automatica::transfer_function::Tf;
use automatica::{polynomial::Poly, Eval};

use num_complex::Complex;

fn main() {
    let tf = Tf::new(
        Poly::new_from_coeffs(&[-0.75, 0.25]),
        Poly::new_from_coeffs(&[0.75, 0.75, 1.]),
    );

    println!("T:\n{}", tf);

    let c = tf.eval(Complex::new(0., 0.9));
    println!(
        "{}\n{}dB, {}°",
        c,
        20. * c.norm().log(10.),
        c.arg().to_degrees()
    );
}
