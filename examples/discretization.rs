#[macro_use]
extern crate automatica;

use automatica::{
    linear_system::discrete::DiscreteTime,
    signals::{continuous, discrete},
    Discretization, Seconds, Ss, Tf,
};

fn main() {
    let num = poly!(4.);
    let den = poly!(4., 1., 2.);
    let g = Tf::new(num, den);
    println!("Transfer function:\n{}\n", g);

    let sys = Ss::new_observability_realization(&g).unwrap();
    println!("State space representation:\n{}\n", &sys);
    let x0 = [0., 0.];
    let steps = 250;
    let sampling_time = 0.1;

    let it = sys.rk2(continuous::step(1., 1), &x0, Seconds(sampling_time), steps);

    // Choose between
    // Discretization::ForwardEuler
    // Discretization::BackwardEuler
    // Discretization::Tustin
    let method = Discretization::Tustin;
    let disc = sys.discretize(sampling_time, method).unwrap();
    println!("Discretization method: {:?}", method);
    println!("{}", &disc);

    let te = disc.evolution_fn(steps, discrete::step_vec(1., 0, 1), &[0., 0.]);

    println!("Time; continuous - step; discrete");
    for (i, j) in it.zip(te).step_by(10) {
        print!("{:>4.1}; {:>10.5} - ", i.time(), i.output()[0],);
        println!("{:>4.1}; {:>8.5}", j.time(), j.output()[0],);
    }
}
