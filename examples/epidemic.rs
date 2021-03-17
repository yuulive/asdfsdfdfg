extern crate au;

use au::{signals::discrete, Ssd};

// Boyd S., Introduction to Applied Linear Algebra, Cambridge University Press 2018, par. 9.3
fn main() {
    let susceptible = 0;
    let infected = 1;
    let recovered = 2;
    let deceased = 3;

    let a = [
        0.95, 0.04, 0.00, 0.00, // New susceptible.
        0.05, 0.85, 0.00, 0.00, // New infected.
        0.00, 0.10, 1.00, 0.00, // New recovered.
        0.00, 0.01, 0.00, 1.00, // New deceased.
    ];
    // No external input.
    let b = [0., 0., 0., 0.];
    // Replicate states to outputs.
    let c = [
        1., 0., 0., 0., //
        0., 1., 0., 0., //
        0., 0., 1., 0., //
        0., 0., 0., 1., //
    ];
    // No external input.
    let d = [0., 0., 0., 0.];

    let sys = Ssd::new_from_slice(4, 1, 4, &a, &b, &c, &d);
    println!("{}", &sys);

    let steps = 200;
    let initial_state = [1., 0., 0., 0.];
    let te = sys.evolution_fn(steps, discrete::zero(1), &initial_state);

    println!("Time Susceptible Infected Recovered Deceased");
    for step in te.step_by(10) {
        println!(
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
            step.time(),
            step.output()[susceptible],
            step.output()[infected],
            step.output()[recovered],
            step.output()[deceased],
        );
    }
}
