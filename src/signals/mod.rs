//! Collection of commons input signals.

pub mod continuous {
    //! Collection of continuous signals.
    use crate::units::{RadiansPerSecond, Seconds};
    use num_traits::Float;

    /// Zero input function
    ///
    /// # Arguments
    ///
    /// * `size` - Output size
    pub fn zero<T: Float>(size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |_| vec![T::zero(); size]
    }

    /// Impulse function
    ///
    /// # Arguments
    ///
    /// * `k` - Impulse size
    /// * `o` - Impulse time
    /// * `size` - Output size
    pub fn impulse<T: Float>(k: T, o: Seconds<T>, size: usize) -> impl Fn(Seconds<T>) -> Vec<T> {
        move |t| {
            if t == o {
                vec![k; size]
            } else {
                vec![T::zero(); size]
            }
        }
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
            relative_eq!(0., zero(1)(Seconds(s))[0])
        }

        #[test]
        fn impulse_input() {
            let imp = impulse(10., Seconds(1.), 1);
            assert_relative_eq!(0., imp(Seconds(0.5))[0]);
            assert_relative_eq!(10., imp(Seconds(1.))[0]);
        }

        #[quickcheck]
        fn step_input(s: f64) -> bool {
            relative_eq!(3., step(3., 1)(Seconds(s))[0])
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
    use num_traits::Float;

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
    /// * `time` - Time at which step occurs
    pub fn step<T: Float>(k: T, time: usize) -> impl Fn(usize) -> T {
        move |t| {
            if t < time {
                T::zero()
            } else {
                k
            }
        }
    }

    /// Step function
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `time` - Time at which step occurs
    /// * `size` - Output size
    pub fn step_vec<T: Float>(k: T, time: usize, size: usize) -> impl Fn(usize) -> Vec<T> {
        move |t| {
            if t < time {
                vec![T::zero(); size]
            } else {
                vec![k; size]
            }
        }
    }

    /// Impulse function at given time
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `time` - Impulse time
    pub fn impulse<T: Float>(k: T, time: usize) -> impl Fn(usize) -> T {
        move |t| {
            if t == time {
                k
            } else {
                T::zero()
            }
        }
    }

    /// Impulse function at given time
    ///
    /// # Arguments
    ///
    /// * `k` - Step size
    /// * `time` - Impulse time
    /// * `size` - Output size
    pub fn impulse_vec<T: Float>(k: T, time: usize, size: usize) -> impl Fn(usize) -> Vec<T> {
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
            relative_eq!(0., zero(1)(s)[0])
        }

        #[quickcheck]
        fn step_single_input(s: f32) -> bool {
            let f = step(s, 2);
            relative_eq!(0., f(1)) && relative_eq!(s, f(2))
        }

        #[quickcheck]
        fn step_input(s: usize) -> bool {
            let f = step_vec(3., 1, 1);
            if s == 0 {
                relative_eq!(0., f(s)[0])
            } else {
                relative_eq!(3., f(s)[0])
            }
        }

        #[test]
        fn step_input_at_zero() {
            let f = step_vec(3., 1, 1);
            assert_relative_eq!(0., f(0)[0]);
        }

        #[quickcheck]
        fn impulse_single_input(i: f32) -> bool {
            let f = impulse(i, 2);
            relative_eq!(0., f(1)) && relative_eq!(i, f(2))
        }

        #[test]
        fn impulse_input() {
            let mut out: Vec<_> = (0..20).map(|t| impulse_vec(10., 15, 1)(t)[0]).collect();
            assert_relative_eq!(10., out[15]);
            out.remove(15);
            assert!(out.iter().all(|&o| o == 0.))
        }
    }
}
