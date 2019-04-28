extern crate automatica;

use automatica::linear_system::Ss;
use nalgebra::DMatrix;

fn main() {
    let a = DMatrix::from_row_slice(2, 2, &[-1., 1., -1., 0.25]);
    let b = DMatrix::from_row_slice(2, 1, &[1., 0.25]);
    let c = DMatrix::from_row_slice(2, 2, &[0., 1., -1., 1.]);
    let d = DMatrix::from_row_slice(2, 1, &[0., 1.]);

    let sys = Ss::new(a, b, c, d);
    let poles = sys.poles();

    println!("A: {}", &sys.a());
    println!("B: {}", &sys.b());
    println!("C: {}", &sys.c());
    println!("D: {}", &sys.d());
    println!("poles: {}", poles);
}
