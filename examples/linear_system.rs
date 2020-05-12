extern crate automatica;

use num_complex::Complex;

use automatica::{signals::continuous, Seconds, Ss, TfMatrix};

#[allow(clippy::many_single_char_names)]
#[allow(clippy::non_ascii_literal)]
fn main() {
    let a = [-1., 1., -1., 0.25];
    let b = [1., 0.25];
    let c = [0., 1., -1., 1.];
    let d = [0., 1.];

    let sys = Ss::new_from_slice(2, 1, 2, &a, &b, &c, &d);
    let poles = sys.poles();

    println!("{}", &sys);
    println!("Poles: {:.2}, {:.2}", poles[0], poles[1]);

    println!("\nUnitary step response:");
    let step = continuous::step(1., 1);

    let rk2 = sys.rk2(&step, &[0., 0.], Seconds(0.1), 150);
    println!("rk2 stationary values:");
    let last_rk2 = rk2.last().unwrap();
    print_step(&last_rk2);
    // Change to 'true' to print the result
    if false {
        for i in sys.rk2(&step, &[0., 0.], Seconds(0.1), 150) {
            print_step(&i);
        }
    }

    let rk4 = sys.rk4(&step, &[0., 0.], Seconds(0.1), 150);
    println!("rk4 stationary values:");
    let last_rk4 = rk4.last().unwrap();
    print_step(&last_rk4);
    // Change to 'true' to print the result
    if false {
        for i in sys.rk4(&step, &[0., 0.], Seconds(0.1), 150) {
            print_step(&i);
        }
    }

    let rkf45 = sys.rkf45(&step, &[0., 0.], Seconds(0.1), Seconds(16.), 1e-4);
    println!("rkf45 stationary values:");
    let last_rkf45 = rkf45.last().unwrap();
    print_step_with_error(&last_rkf45);
    // Change to 'true' to print the result
    if false {
        for i in sys.rkf45(&step, &[0., 0.], Seconds(0.1), Seconds(16.), 1e-4) {
            print_step_with_error(&i);
        }
    }

    let radau = sys.radau(&step, &[0., 0.], Seconds(0.1), 150, 1e-4);
    println!("radau stationary values:");
    let last_radau = radau.last().unwrap();
    print_step(&last_radau);
    // Change to 'true' to print the result
    if false {
        for i in sys.radau(&step, &[0., 0.], Seconds(0.1), 150, 1e-4) {
            print_step(&i);
        }
    }

    let u = 0.0;
    println!("\nEquilibrium for u = {}", u);
    let eq = sys.equilibrium(&[u]).unwrap();
    println!("x = {:?}\ny = {:?}", eq.x(), eq.y());

    println!("\nTransform linear system into a transfer function");
    let tf_matrix = TfMatrix::from(sys);
    println!("Tf:\n{}", tf_matrix);

    println!("\nEvaluate transfer function in ω = 0.9");
    let u = vec![Complex::new(0.0, 0.9)];
    let y = tf_matrix.eval(&u);
    let y1 = &y[0];
    let y2 = &y[1];
    println!(
        "u = {:.2} => y = [{:.2}, {:.2}] = [{:.2}<{:+.2}, {:.2}<{:+.2}]",
        &u[0],
        y1,
        y2,
        y1.norm(),
        y1.arg().to_degrees(),
        y2.norm(),
        y2.arg().to_degrees()
    );
}

fn print_step(s: &automatica::linear_system::solver::Step<f64>) {
    println!(
        "Time: {:.2}; state: [{:.4}; {:.4}]; output: [{:.4}; {:.4}]",
        s.time(),
        s.state()[0],
        s.state()[1],
        s.output()[0],
        s.output()[1]
    );
}

fn print_step_with_error(s: &automatica::linear_system::solver::StepWithError<f64>) {
    println!(
        "Time: {:.2}; state: [{:.4}; {:.4}]; output: [{:.4}; {:.4}]; error: {:.5}",
        s.time(),
        s.state()[0],
        s.state()[1],
        s.output()[0],
        s.output()[1],
        s.error()
    );
}
