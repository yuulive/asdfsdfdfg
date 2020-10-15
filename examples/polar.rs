#[macro_use]
extern crate automatica;

use automatica::{
    plots::polar::{Plotter, Polar},
    Poly, RadiansPerSecond, Tf, Tfz,
};

fn main() {
    let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));

    println!("T:\n{}\n", tf);

    let p = Polar::new(tf, RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
    for g in p {
        println!(
            "({:.3}) => mag: {:.3}, phase: {:.3}",
            g.output(),
            g.magnitude(),
            g.phase()
        );
    }

    let k = 0.5;
    let tfz = Tfz::new(poly!(1. - k), poly!(-k, 1.));
    println!("T:\n{}\n", tfz);
    let pz = Polar::new_discrete(tfz, RadiansPerSecond(0.01), 0.1);
    for g in pz {
        println!(
            "({:.3}) => mag: {:.3}, phase: {:.3}",
            g.output(),
            g.magnitude(),
            g.phase()
        );
    }
}
