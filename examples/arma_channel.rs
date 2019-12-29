extern crate automatica;

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use automatica::{poly, Tfz};

fn main() {
    let tf = Tfz::new(poly!(1.), poly!(1., 0.5));

    println!("T:\n{}", tf);

    let (send, recv) = mpsc::channel();
    thread::spawn(move || {
        for i in &[0.1, 0.3, 0.6, 0.8, 1.0] {
            send.send(*i).unwrap();
            thread::sleep(Duration::from_millis(50));
        }
    });

    let arma = tf.arma_iter(recv.iter());
    for y in arma {
        println!("{}", y);
    }
    // y = dsimul(tf2ss(1/(1+0.5*%z)), [0.1, 0.3, 0.6, 0.8, 1.0])
}
