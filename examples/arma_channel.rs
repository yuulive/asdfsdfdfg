extern crate automatica;

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use automatica::{poly, Tfz};

fn main() {
    let tf = Tfz::new(poly!(1.), poly!(1., 0.5));

    println!("T:\n{}\n", tf);

    let input = &[0.1, 0.3, 0.6, 0.8, 1.0];
    let (send, recv) = mpsc::channel();
    thread::spawn(move || {
        for i in input {
            send.send(*i).unwrap();
            // Simulate sensor timing.
            thread::sleep(Duration::from_millis(50));
        }
    });

    println!("Input sent from the channel:");
    for u in input {
        print!("{:.2} ", u);
    }
    println!();

    println!("Transformed values:");
    let arma = tf.arma_iter(recv);
    for y in arma {
        print!("{:.2} ", y);
    }
    println!();
    // y = dsimul(tf2ss(1/(1+0.5*%z)), [0.1, 0.3, 0.6, 0.8, 1.0])
}
