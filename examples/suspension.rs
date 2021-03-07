use num_traits::One;

use automatica::{Discretization, Poly, Seconds, Ss, Tf, TfMatrix, Tfz};

#[allow(non_snake_case)]
fn main() {
    // Linear system describing a car suspension.
    let ms = 400.0; // kg
    let mu = 50.0; // kg
    let ks = 2.0e4; // N/m
    let ku = 2.5e5; // N/m
    let cs = 2.0e3; // Ns/m

    let A = [
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        -ks / ms,
        ks / ms,
        -cs / ms,
        cs / ms,
        ks / mu,
        -(ks + ku) / mu,
        cs / mu,
        -cs / mu,
    ];
    let B = [0., 0., 0., 0., 0., 1. / ms, ku / mu, -1. / mu];
    let C = [-ks / ms, ks / ms, -cs / ms, cs / ms, 0., ku, 0., 0.];
    let D = [0., 1. / ms, -ku, 0.];

    let sys = Ss::new_from_slice(4, 2, 2, &A, &B, &C, &D);

    println!("Linear system:\n{}", &sys);

    let G = TfMatrix::from(sys);
    let G11 = Tf::new(G[[0, 0]].clone(), G.den());
    let G12 = Tf::new(G[[0, 1]].clone(), G.den());

    println!("\nTranfer functions:\n{}", &G);

    let ur = 1.0e5;
    let R = Tf::new(
        ur * Poly::new_from_roots(&[-1.0, -1.]),
        Poly::new_from_coeffs(&[1., 10.]) * Poly::new_from_coeffs(&[1., 10.]),
    );
    println!("\nRegulator:\n{}", &R);

    let tau = 0.01;
    let Ga = Tf::new(Poly::one(), Poly::new_from_coeffs(&[1., tau]));
    println!("\nActuator:\n{}", &Ga);

    // Loop function;
    let L = &R * &Ga * G12;
    println!("\nLoop function:\n{}", &L);
    let one = Tf::new(Poly::one(), Poly::one());
    let M = G11 / (one + L);

    println!("\nFinal response:\n{}", &M);

    println!("\nPoles:");
    for p in M.complex_poles() {
        println!("{:.3}", p);
    }

    let tfzR = R
        .discretize(Seconds(5.0e-3), Discretization::Tustin)
        .normalize();
    println!("\nDiscrete regulator by Tustin method:\n{:.3}", tfzR);

    let ssR = Ss::new_observability_realization(&R).unwrap();
    let sdR = ssR.discretize(5.0e-3, Discretization::Tustin).unwrap();
    let disc_sysR = Tfz::<f64>::new_from_siso(&sdR).unwrap();
    println!(
        "\nDiscrete regulator by Tustin method, discretizing linear sys:\n{:.3}",
        disc_sysR
    );
}
