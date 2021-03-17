extern crate au;

use au::{signals::continuous, Seconds, Ss, Tf};

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

    let tr = Tf::<f64>::new_from_siso(&sys).unwrap();
    println!("{}", &tr);

    let poles = tr.complex_poles();
    println!("\nPoles: {:.2}, {:.2}", poles[0], poles[1]);
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
    let stiff_poles = stiff_sys.poles();
    println!("Poles: {:.2}, {:.2}", stiff_poles[0], stiff_poles[1]);

    // Free movement.
    let zero_input = continuous::zero(1);
    let x0 = &[1., 0.];

    // Solvers.
    let rk2 = stiff_sys.rk2(&zero_input, x0, Seconds(0.1), 5);
    let rkf45 = stiff_sys.rkf45(&zero_input, x0, Seconds(0.1), Seconds(5.), 1e-3);
    let radau = stiff_sys.radau(&zero_input, x0, Seconds(0.1), 70, 1e-3);

    println!(
        "Rk2 number of steps before diverging: {}",
        rk2.take_while(|y| y.output()[0].abs() < 1000.).count()
    );
    println!("Rkf45 number of steps: {}", rkf45.count());
    println!(
        "Radau stationary output: {:.5}",
        radau.last().unwrap().output()[0]
    );
}
