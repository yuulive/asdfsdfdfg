extern crate automatica;

use automatica::linear_system::Ss;
use automatica::transfer_function::Tf;

use std::convert::TryFrom;

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

    println!("\n{:?}", tr.complex_poles());
}
