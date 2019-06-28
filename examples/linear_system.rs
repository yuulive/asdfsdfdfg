extern crate automatica;

use automatica::linear_system::Ss;
use automatica::transfer_function::TfMatrix;
use automatica::Eval;

use num_complex::Complex;

fn main() {
    let a = [-1., 1., -1., 0.25];
    let b = [1., 0.25];
    let c = [0., 1., -1., 1.];
    let d = [0., 1.];

    let sys = Ss::new_from_slice(2, 1, 2, &a, &b, &c, &d);
    let poles = sys.poles();

    println!("{}", &sys);
    println!("poles: {:?}", poles);

    let u = 0.0;
    println!("\nEquilibrium for u={}", u);
    let eq = sys.equilibrium(&[u]).unwrap();
    println!("x: {:?}\ny: {:?}", eq.x(), eq.y());

    println!("\nTransform linear system into a transfer function");
    let tf_matrix = TfMatrix::from(sys);
    println!("Tf:{}", tf_matrix);

    println!("\nEvaluate transfer function in ω = 0.9");
    let u = vec![Complex::new(0.0, 0.9)];
    let y = tf_matrix.eval(&u);
    println!("u:{:?}\ny:{:?}", &u, &y);

    println!(
        "y:{:?}",
        &y.iter().map(|x| (x.norm(),x.arg().to_degrees())).collect::<Vec<_>>()
    );
}
