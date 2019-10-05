extern crate automatica;

use crate::automatica::polynomial::Poly;

fn main() {
    let p = Poly::new_from_coeffs(&[1., -2., 3.]);
    println!("{}", p);

    let p2 = Poly::new_from_coeffs(&[1., 0., 3., 0., -12.]);
    println!("{}", p2);

    let p3 = Poly::new_from_roots(&[1., -2., 3.]);
    println!("{}", p3);

    println!("{}", 2. + p3 + 2.);

    println!("\nTartaglia's triangle:\n1");
    let m = Poly::new_from_coeffs(&[1, 1]);
    let mut tot = m.clone();
    println!("{}", m);
    for _ in 0..5 {
        tot = &tot * &m;
        println!("{}", tot);
    }
}
