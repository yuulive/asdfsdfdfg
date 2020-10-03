#[macro_use]
extern crate automatica;

use automatica::{plots::polar::Polar, Poly, RadiansPerSecond, Tf};

fn main() {
    let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));

    println!("T:\n{}\n", tf);

    let p = tf.polar(RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    for g in p {
        println!(
            "({:.3}) => mag: {:.3}, phase: {:.3}",
            g.output(),
            g.magnitude(),
            g.phase()
        );
    }
}
