//! Collection of commons input signals.

use num_traits::Float;

use crate::units::{RadiansPerSecond, Seconds};

pub mod continuous {
    //! Collection of continuous signals.
    use super::*;

    /// Zero input function
    ///
    /// # Arguments
    ///
    /// * `size` - Output size
    pub fn zero<T: Float>(size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |_| vec![T::zero(); size]
    }

    /// Step function
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `size` - Output size
    pub fn step<T: Float>(k: T, size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |_| vec![k; size]
    }

    /// Sine input (single input single output).
    ///
    /// `sin(omega*t - phase)`
    ///
    /// # Arguments
    ///
    /// * `a` - sine amplitude
    /// * `omega` - sine pulse in radians per second
    /// * `phi` - sine phase in radians
    pub fn sin_siso<T: Float>(
        a: T,
        omega: RadiansPerSecond<T>,
        phi: T,
    ) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |t| vec![a * T::sin(omega.0 * t.0 - phi)]
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::f64::consts::PI;

        #[quickcheck]
        fn zero_input(s: f64) -> bool {
            0. == zero(1)(Seconds(s))[0]
        }

        #[quickcheck]
        fn step_input(s: f64) -> bool {
            3. == step(3., 1)(Seconds(s))[0]
        }

        #[quickcheck]
        fn sin_input(t: f64) -> bool {
            let sine = sin_siso(1., RadiansPerSecond(0.5), 0.)(Seconds(t))[0];
            let traslated_sine = sin_siso(1., RadiansPerSecond(0.5), PI)(Seconds(t))[0];
            relative_eq!(sine, -traslated_sine, max_relative = 1e-10)
        }
    }
}

pub mod discrete {
    //! Collection of discrete signals.
    use super::*;

    /// Zero input function
    ///
    /// # Arguments
    ///
    /// * `size` - Output size
    pub fn zero<T: Float>(size: usize) -> impl Fn(usize) -> Vec<T> {
        move |_| vec![T::zero(); size]
    }

    /// Step function
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `size` - Output size
    pub fn step<T: Float>(k: T, size: usize) -> impl Fn(usize) -> Vec<T> {
        move |_| vec![k; size]
    }

    /// Impulse function at given time
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `time` - Impulse time
    /// * `size` - Output size
    pub fn impulse<T: Float>(k: T, time: usize, size: usize) -> impl Fn(usize) -> Vec<T> {
        move |t| {
            if t == time {
                vec![k; size]
            } else {
                vec![T::zero(); size]
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[quickcheck]
        fn zero_input(s: usize) -> bool {
            0. == zero(1)(s)[0]
        }

        #[quickcheck]
        fn step_input(s: usize) -> bool {
            3. == step(3., 1)(s)[0]
        }

        #[test]
        fn impulse_input() {
            let mut out: Vec<_> = (0..20).map(|t| impulse(10., 15, 1)(t)[0]).collect();
            assert_eq!(10., out[15]);
            out.remove(15);
            assert!(out.iter().all(|&o| o == 0.))
        }
    }
}
