extern crate automatica;

use automatica::{signals::continuous, Seconds, Ss, Tf};

use std::convert::TryFrom;

#[allow(clippy::many_single_char_names)]
fn main() {
    // Mass (m), spring (k), dumper (f)
    // external force (u)
    // position (x1), speed (x2)
    // position (y)
    let m = 1.;
    let k = 1.;
    let f = 1.;

    let a = [0., 1., -k / m, -f / m];
    let b = [0., 1. / m];
    let c = [1., 0.];
    let d = [0.];

    let sys = Ss::new_from_slice(2, 1, 1, &a, &b, &c, &d);
    println!("{}", &sys);

    let tr = Tf::try_from(sys).unwrap();
    println!("{}", &tr);

    println!("\nPoles:\n{:?}", tr.complex_poles());
    //let radau = sys.radau(|t| vec![t.cos()], &[0., 0.], 0.1, 200, 1e-3);

    // Make a stiff system.
    let k_stiff = 1000.;
    let f_stiff = 1001.;
    let a_stiff = [0., 1., -k_stiff / m, -f_stiff / m];
    let stiff_sys = Ss::new_from_slice(2, 1, 1, &a_stiff, &b, &c, &d);
    println!(
        "\nThe system becomes stiff with k={} and f={}",
        k_stiff, f_stiff
    );

    // Free movement.
    let null_input = continuous::zero(1);
    let x0 = &[1., 0.];

    // Solvers.
    let rk2: Vec<_> = stiff_sys.rk2(&null_input, x0, Seconds(0.1), 5).collect();
    let rkf45: Vec<_> = stiff_sys
        .rkf45(&null_input, x0, Seconds(0.1), Seconds(5.), 1e-3)
        .collect();
    let radau: Vec<_> = stiff_sys
        .radau(&null_input, x0, Seconds(0.1), 70, 1e-3)
        .collect();

    println!(
        "Rk2 number of steps before diverging: {}",
        rk2.iter()
            .take_while(|y| y.output()[0].abs() < 1000.)
            .count()
    );
    println!("Rkf45 number of steps: {}", rkf45.len());
    println!("Radau stationary: {}", radau.last().unwrap().output()[0]);
}
