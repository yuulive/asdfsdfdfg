extern crate au;

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use au::{poly, Tfz};

fn main() {
    let tf = Tfz::new(poly!(1.), poly!(1., 0.5));

    println!("T:\n{}\n", tf);

    let input = &[0.1, 0.3, 0.6, 0.8, 1.0];
    let (sensor_tx, sensor_rx) = mpsc::channel();
    thread::spawn(move || {
        for i in input {
            sensor_tx.send(*i).unwrap();
            // Simulate sensor timing.
            thread::sleep(Duration::from_millis(50));
        }
    });

    let (actuator_tx, actuator_rx) = mpsc::channel();
    thread::spawn(move || {
        for i in actuator_rx {
            print!("{:.2} ", i);
        }
    });

    println!("Input sent from the channel:");
    for u in input {
        print!("{:.2} ", u);
    }
    println!();

    println!("Transformed values:");
    let arma = tf.arma_iter(sensor_rx);
    for y in arma {
        actuator_tx.send(y).unwrap();
    }
    println!();
    // y = dsimul(tf2ss(1/(1+0.5*%z)), [0.1, 0.3, 0.6, 0.8, 1.0])
}
