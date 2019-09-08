extern crate automatica;

use automatica::{
    linear_system::{
        discrete::{Discrete, Discretization},
        Ss,
    },
    polynomial::Poly,
    transfer_function::Tf,
    units::Seconds,
};

fn main() {
    let num = Poly::new_from_coeffs(&[4.]);
    let den = Poly::new_from_coeffs(&[4., 1., 2.]);
    let g = Tf::new(num, den);
    println!("{}", g);

    let step1 = |_: Seconds| vec![1.];
    let step2 = |_: usize| vec![1.];

    let sys = Ss::from(g);
    println!("{}", &sys);
    let x0 = [0., 0.];
    let steps = 250;
    let it = sys.rk2(step1, &x0, Seconds(0.1), steps);
    if false {
        for i in it {
            println!("{:.1};{:.5}", i.time().0, i.output()[0],);
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

    let te = disc.time_evolution(steps, step2, &[0., 0.]);
    if false {
        for i in te {
            println!("{:.1};{:.5}", i.time(), i.output()[0],);
        }
    }
}
