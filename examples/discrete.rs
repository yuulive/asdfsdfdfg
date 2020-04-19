extern crate automatica;

use automatica::{signals::discrete, Ssd};

fn main() {
    let a = [0.2, 0., 0., 0.6, 0.15, 0., 0., 0.8, 0.08];
    let b = [1., 0., 0.];
    let c = [0., 0., 0.9];
    let d = [0.];

    let sys = Ssd::new_from_slice(3, 1, 1, &a, &b, &c, &d);
    println!("{}", &sys);

    let input = 50.;
    let te = sys.evolution_fn(8, discrete::step_vec(input, 0, 1), &[0., 0., 0.]);

    let last_step = te.last().unwrap();
    println!("{:?}", last_step);

    let last_output = last_step.output()[0];
    let gain = last_output / input;
    println!("Gain: {:.3} / {} = {:.3}", last_output, input, gain);
}
