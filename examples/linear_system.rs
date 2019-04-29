extern crate automatica;

use automatica::linear_system::{self, Ss};
use nalgebra::DMatrix;

fn main() {
    let a = DMatrix::from_row_slice(2, 2, &[-1., 1., -1., 0.25]);
    let b = DMatrix::from_row_slice(2, 1, &[1., 0.25]);
    let c = DMatrix::from_row_slice(2, 2, &[0., 1., -1., 1.]);
    let d = DMatrix::from_row_slice(2, 1, &[0., 1.]);

    let sys = Ss::new(a, b, c, d);
    let poles = sys.poles();

    println!("{}", &sys);
    println!("poles: {}", poles);

    let eq = sys.equilibrium(&[0.]);
    println!("{}", eq.unwrap());

    let a = DMatrix::from_row_slice(3,3,&[3.,1.,5.,3.,3.,1.,4.,6.,4.]);
    let (p, B) = linear_system::leverrier(&a);
    println!("A: {}\np: {}\nB: {:?}", &a, &p, &B);
}
