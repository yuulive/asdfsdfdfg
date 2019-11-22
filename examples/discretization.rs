#[macro_use]
extern crate automatica;

use std::convert::TryFrom;

use automatica::{
    linear_system::{
        discrete::{Discrete, Discretization},
        Ss,
    },
    signals::{continuous, discrete},
    transfer_function::Tf,
    units::Seconds,
};

fn main() {
    let num = poly!(4.);
    let den = poly!(4., 1., 2.);
    let g = Tf::new(num, den);
    println!("{}", g);

    let sys = Ss::try_from(g).unwrap();
    println!("{}", &sys);
    let x0 = [0., 0.];
    let steps = 250;
    let it = sys.rk2(continuous::step(1., 1), &x0, Seconds(0.1), steps);
    if false {
        for i in it {
            println!("{:.1};{:.5}", i.time(), i.output()[0],);
        }
    }

    // Choose between
    // Discretization::ForwardEuler
    // Discretization::BackwardEuler
    // Discretization::Tustin
    let method = Discretization::Tustin;
    let disc = sys.discretize(0.1, method).unwrap();
    println!("Discretization method: {:?}", method);
    println!("{}", &disc);

    let te = disc.time_evolution(steps, discrete::step(1., 1), &[0., 0.]);
    if false {
        for i in te {
            println!("{:.1};{:.5}", i.time(), i.output()[0],);
        }
    }
}
