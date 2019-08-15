extern crate automatica;

use automatica::linear_system::{discrete::Discrete, Ss};

fn main() {
    let a = [0.2, 0., 0., 0.6, 0.15, 0., 0., 0.8, 0.08];
    let b = [1., 0., 0.];
    let c = [0., 0., 0.9];
    let d = [0.];

    let sys = Ss::new_from_slice(3, 1, 1, &a, &b, &c, &d);
    println!("{}", &sys);

    let te = sys.time_evolution(8, |_| vec![50.], &[0.,0.,0.]);

    println!("{:?}", te.last().unwrap());
}
