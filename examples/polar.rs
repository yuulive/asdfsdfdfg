extern crate automatica;

use automatica::plots::polar::PolarPlot;
use automatica::polynomial::Poly;
use automatica::transfer_function::Tf;

fn main() {
    let tf = Tf::new(
        Poly::new_from_coeffs(&[5.]),
        Poly::new_from_roots(&[-1., -10.]),
    );

    println!("T:\n{}", tf);

    let p = tf.polar(0.1, 10.0, 0.1);
    for g in p {
        println!(
            "r: {:.3}, i: {:.3}, f: {:.3}, m: {:.3}",
            g.real(),
            g.imag(),
            g.phase(),
            g.magnitude()
        );
    }
}
