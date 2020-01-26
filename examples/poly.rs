#[macro_use]
extern crate automatica;

use crate::automatica::Poly;

fn main() {
    let p = poly!(1., -2., 3.);
    println!("{}", p);

    let p2 = poly!(1., 0., 3., 0., -12.);
    println!("{}", p2);

    let p3 = Poly::new_from_roots(&[1., -2., 3.]);
    println!("{}", p3);

    println!("{}", 2. + p3 + 2.);

    println!("\nTartaglia's triangle:\n1");
    let m = poly!(1, 1);
    let mut tot = m.clone();
    println!("{}", m);
    for _ in 0..5 {
        tot = &tot * &m;
        println!("{}", tot);
    }

    println!("\nWilkinson's polynomial:");
    println!("p(x) = (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10)*");
    println!("      *(x-11)(x-12)(x-13)(x-14)(x-15)(x-16)(x-17)(x-18)(x-19)(x-20)");
    let wp = Poly::new_from_roots(&[
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
    ]);
    println!("p(x) = {}\n", wp);
    let mut roots = wp.iterative_roots();
    roots.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());
    println!("p(x) roots with Aberth-Ehrlich Method:");
    for r in roots {
        println!("{}", r);
    }
    let mut roots2 = wp.complex_roots();
    roots2.sort_by(|&x, &y| x.re.partial_cmp(&y.re).unwrap());
    println!("\np(x) roots with eigenvalue decomposition:");
    for r in roots2 {
        println!("{}", r);
    }
}
