extern crate automatica;

use automatica::linear_system;
use automatica::polynomial::MP;
use nalgebra::DMatrix;

fn main() {
    println!("\nExample of LeVerrier algorithm (Wikipedia)");
    let t = DMatrix::from_row_slice(3, 3, &[3., 1., 5., 3., 3., 1., 4., 6., 4.]);
    let (p, poly_matrix) = linear_system::leverrier(&t);
    println!("T: {}\np: {}\n", &t, &p);
    println!("B: {}", &poly_matrix);

    let mp = MP::from(poly_matrix);
    println!("mp {}", &mp);
}
