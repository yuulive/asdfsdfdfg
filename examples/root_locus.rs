extern crate automatica;

use automatica::{poly, Poly, Tf};

fn main() {
    let tf = Tf::new(poly!(1.0_f32), Poly::new_from_roots(&[-1., -2.]));

    println!("T:\n{}\n", tf);

    let loci = tf.root_locus_plot(0.1, 1.0, 0.05);
    for locus in loci {
        let out = locus.output();
        println!(
            "k: {1:.0$}, roots: {2:.0$}, {3:.0$}",
            2,
            locus.k(),
            out[0],
            out[1]
        );
    }
}
