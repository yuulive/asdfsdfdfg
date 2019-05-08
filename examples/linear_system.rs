extern crate automatica;

use automatica::linear_system::Ss;
use automatica::transfer_function::TfMatrix;
use automatica::Eval;
use nalgebra::{Complex, DMatrix, DVector};

fn main() {
    let a = DMatrix::from_row_slice(2, 2, &[-1., 1., -1., 0.25]);
    let b = DMatrix::from_row_slice(2, 1, &[1., 0.25]);
    let c = DMatrix::from_row_slice(2, 2, &[0., 1., -1., 1.]);
    let d = DMatrix::from_row_slice(2, 1, &[0., 1.]);

    let sys = Ss::new(&a, &b, &c, &d);
    let poles = sys.poles();

    println!("{}", &sys);
    println!("poles: {}", poles);

    println!("Equilibrium for u=0");
    let eq = sys.equilibrium(&[0.]);
    println!("{}", eq.unwrap());

    println!("Transform linear system into a transfer function");
    let tf_matrix = TfMatrix::from(sys);
    println!("Tf:{}", tf_matrix);

    println!("\nEvaluate transfer function in ω = 0.9");
    let u = DVector::from_element(1, Complex::new(0.0, 0.9));
    let ynum = tf_matrix.eval(&u);
    println!("u:{}\ny:{}", &u, &ynum);
}
