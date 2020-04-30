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
    println!("poles:\n{:?}", poles);

    println!("\nStep response:");
    let step = continuous::step(1., 1);

    let rk2 = sys.rk2(&step, &[0., 0.], Seconds(0.1), 150);
    println!("rk2 stationary values: {:?}", rk2.last().unwrap());
    // Change to 'true' to print the result
    if false {
        for i in sys.rk2(&step, &[0., 0.], Seconds(0.1), 150) {
            println!(
                "{};{};{};{};{}",
                i.time(),
                i.state()[0],
                i.state()[1],
                i.output()[0],
                i.output()[1]
            );
        }
    }

    let rk4 = sys.rk4(&step, &[0., 0.], Seconds(0.1), 150);
    println!("rk4 stationary values: {:?}", rk4.last().unwrap());
    // Change to 'true' to print the result
    if false {
        for i in sys.rk4(&step, &[0., 0.], Seconds(0.1), 150) {
            println!(
                "{};{};{};{};{}",
                i.time(),
                i.state()[0],
                i.state()[1],
                i.output()[0],
                i.output()[1]
            );
        }
    }

    let rkf45 = sys.rkf45(&step, &[0., 0.], Seconds(0.1), Seconds(16.), 1e-4);
    println!("rkf45 stationary values: {:?}", rkf45.last().unwrap());
    // Change to 'true' to print the result
    if false {
        for i in sys.rkf45(&step, &[0., 0.], Seconds(0.1), Seconds(16.), 1e-4) {
            println!(
                "{};{};{};{};{};{}",
                i.time(),
                i.state()[0],
                i.state()[1],
                i.output()[0],
                i.output()[1],
                i.error()
            );
        }
    }

    let radau = sys.radau(&step, &[0., 0.], Seconds(0.1), 150, 1e-4);
    println!("radau stationary values: {:?}", radau.last().unwrap());
    // Change to 'true' to print the result
    if false {
        for i in sys.radau(&step, &[0., 0.], Seconds(0.1), 150, 1e-4) {
            println!(
                "{:>4.1};{:>9.6};{:>9.6};{:>9.6};{:>9.6}",
                i.time(),
                i.state()[0],
                i.state()[1],
                i.output()[0],
                i.output()[1]
            );
        }
    }

    let u = 0.0;
    println!("\nEquilibrium for u={}", u);
    let eq = sys.equilibrium(&[u]).unwrap();
    println!("x:\n{:?}\ny:\n{:?}", eq.x(), eq.y());

    println!("\nTransform linear system into a transfer function");
    let tf_matrix = TfMatrix::from(sys);
    println!("Tf:\n{}", tf_matrix);

    println!("\nEvaluate transfer function in ω = 0.9");
    let u = vec![Complex::new(0.0, 0.9)];
    let y = tf_matrix.eval(&u);
    println!("u:\n{:?}\ny:\n{:?}", &u, &y);

    println!(
        "y:\n{:?}",
        &y.iter()
            .map(|x| (x.norm(), x.arg().to_degrees()))
            .collect::<Vec<_>>()
    );
}
