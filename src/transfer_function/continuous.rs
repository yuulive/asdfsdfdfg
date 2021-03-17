//! # Transfer functions for continuous time systems.
//!
//! Specialized struct and methods for continuous time transfer functions
//! * time delay
//! * initial value and initial derivative value
//! * sensitivity function
//! * complementary sensitivity function
//! * control sensitivity function
//! * root locus plot
//! * bode plot
//! * polar plot
//! * static gain

use nalgebra::RealField;
use num_complex::Complex;
use num_traits::Float;

use std::{cmp::Ordering, marker::PhantomData, ops::Div};

use crate::{
    enums::Continuous,
    plots::{root_locus::RootLocus, Plotter},
    rational_function::Rf,
    transfer_function::TfGen,
    units::Seconds,
};

/// Continuous transfer function
pub type Tf<T> = TfGen<T, Continuous>;

impl<T: Float> Tf<T> {
    /// Time delay for continuous time transfer function.
    /// `y(t) = u(t - tau)`
    /// `G(s) = e^(-tau * s)
    ///
    /// # Arguments
    ///
    /// * `tau` - Time delay
    ///
    /// # Example
    /// ```
    /// use au::{num_complex::Complex, Seconds, Tf};
    /// let d = Tf::delay(Seconds(2.));
    /// assert_eq!(1., d(Complex::new(0., 10.)).norm());
    /// ```
    pub fn delay(tau: Seconds<T>) -> impl Fn(Complex<T>) -> Complex<T> {
        move |s| (-s * tau.0).exp()
    }

    /// System inital value response to step input.
    /// `y(0) = G(s->infinity)`
    ///
    /// # Example
    /// ```
    /// use au::{poly, Tf};
    /// let tf = Tf::new(poly!(4.), poly!(1., 5.));
    /// assert_eq!(0., tf.init_value());
    /// ```
    #[must_use]
    pub fn init_value(&self) -> T {
        let n = self.num().degree();
        let d = self.den().degree();
        match n.cmp(&d) {
            Ordering::Less => T::zero(),
            Ordering::Equal => self.num().leading_coeff() / self.den().leading_coeff(),
            Ordering::Greater => T::infinity(),
        }
    }

    /// System derivative inital value response to step input.
    /// `y'(0) = s * G(s->infinity)`
    ///
    /// # Example
    /// ```
    /// use au::{poly, Tf};
    /// let tf = Tf::new(poly!(1., -3.), poly!(1., 3., 2.));
    /// assert_eq!(-1.5, tf.init_value_der());
    /// ```
    #[must_use]
    pub fn init_value_der(&self) -> T {
        let n = self.num().degree();
        let d = self.den().degree().map(|d| d - 1);
        match n.cmp(&d) {
            Ordering::Less => T::zero(),
            Ordering::Equal => self.num().leading_coeff() / self.den().leading_coeff(),
            Ordering::Greater => T::infinity(),
        }
    }

    /// Sensitivity function for the given controller `r`.
    /// ```text
    ///              1
    /// S(s) = -------------
    ///        1 + G(s)*R(s)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `r` - Controller
    ///
    /// # Example
    /// ```
    /// use au::{poly, Tf};
    /// let g = Tf::new(poly!(1.), poly!(0., 1.));
    /// let r = Tf::new(poly!(4.), poly!(1., 1.));
    /// let s = g.sensitivity(&r);
    /// assert_eq!(Tf::new(poly!(0., 1., 1.), poly!(4., 1., 1.)), s);
    /// ```
    #[must_use]
    pub fn sensitivity(&self, r: &Self) -> Self {
        let n = self.num() * r.num();
        let d = self.den() * r.den();
        Self {
            rf: Rf::new(d.clone(), n + d),
            time: PhantomData,
        }
    }

    /// Complementary sensitivity function for the given controller `r`.
    /// ```text
    ///          G(s)*R(s)
    /// F(s) = -------------
    ///        1 + G(s)*R(s)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `r` - Controller
    ///
    /// # Example
    /// ```
    /// use au::{poly, Tf};
    /// let g = Tf::new(poly!(1.), poly!(0., 1.));
    /// let r = Tf::new(poly!(4.), poly!(1., 1.));
    /// let f = g.compl_sensitivity(&r);
    /// assert_eq!(Tf::new(poly!(4.), poly!(4., 1., 1.)), f);
    /// ```
    #[must_use]
    pub fn compl_sensitivity(&self, r: &Self) -> Self {
        let l = self * r;
        l.feedback_n()
    }

    /// Sensitivity to control function for the given controller `r`.
    /// ```text
    ///            R(s)
    /// Q(s) = -------------
    ///        1 + G(s)*R(s)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `r` - Controller
    ///
    /// # Example
    /// ```
    /// use au::{poly, Tf};
    /// let g = Tf::new(poly!(1.), poly!(0., 1.));
    /// let r = Tf::new(poly!(4.), poly!(1., 1.));
    /// let q = g.control_sensitivity(&r);
    /// assert_eq!(Tf::new(poly!(0., 4.), poly!(4., 1., 1.)), q);
    /// ```
    #[must_use]
    pub fn control_sensitivity(&self, r: &Self) -> Self {
        Self {
            rf: Rf::new(
                r.num() * self.den(),
                r.num() * self.num() + r.den() * self.den(),
            ),
            time: PhantomData,
        }
    }
}

impl<T: Float + RealField> Tf<T> {
    /// System stability. Checks if all poles are negative.
    ///
    /// # Example
    ///
    /// ```
    /// use au::{Poly, Tf};
    /// let tf = Tf::new(Poly::new_from_coeffs(&[1.]), Poly::new_from_roots(&[-1., -2.]));
    /// assert!(tf.is_stable());
    /// ```
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.complex_poles().iter().all(|p| p.re.is_negative())
    }

    /// Root locus for the given coefficient `k`
    ///
    /// # Arguments
    ///
    /// * `k` - Transfer function constant
    ///
    /// # Example
    /// ```
    /// use au::{num_complex::Complex, poly, Poly, Tf};
    /// let l = Tf::new(poly!(1.), Poly::new_from_roots(&[-1., -2.]));
    /// let locus = l.root_locus(0.25);
    /// assert_eq!(Complex::new(-1.5, 0.), locus[0]);
    /// ```
    pub fn root_locus(&self, k: T) -> Vec<Complex<T>> {
        let p = &(self.num() * k) + self.den();
        p.complex_roots()
    }

    /// Create a `RootLocus` plot
    ///
    /// # Arguments
    ///
    /// * `min_k` - Minimum transfer constant of the plot
    /// * `max_k` - Maximum transfer constant of the plot
    /// * `step` - Step between each transfer constant
    ///
    /// `step` is linear.
    ///
    /// # Panics
    ///
    /// Panics if the step is not strictly positive of the minimum transfer constant
    /// is not lower than the maximum transfer constant.
    ///
    /// # Example
    /// ```
    /// use au::{num_complex::Complex, poly, Poly, Tf};
    /// let l = Tf::new(poly!(1.), Poly::new_from_roots(&[-1., -2.]));
    /// let locus = l.root_locus_plot(0.1, 1.0, 0.05).into_iter();
    /// assert_eq!(19, locus.count());
    /// ```
    pub fn root_locus_plot(self, min_k: T, max_k: T, step: T) -> RootLocus<T> {
        RootLocus::new(self, min_k, max_k, step)
    }
}

impl<T> Tf<T> {
    /// Static gain `G(0)`.
    /// Ratio between constant output and constant input.
    /// Static gain is defined only for transfer functions of 0 type.
    ///
    /// Example
    ///
    /// ```
    /// use au::{poly, Tf};
    /// let tf = Tf::new(poly!(4., -3.),poly!(2., 5., -0.5));
    /// assert_eq!(2., tf.static_gain());
    /// ```
    #[must_use]
    pub fn static_gain<'a>(&'a self) -> T
    where
        &'a T: 'a + Div<&'a T, Output = T>,
    {
        &self.num()[0] / &self.den()[0]
    }
}

impl<T: Float> Plotter<T> for Tf<T> {
    /// Evaluate the transfer function at the given value.
    ///
    /// # Arguments
    ///
    /// * `s` - angular frequency at which the function is evaluated
    fn eval_point(&self, s: T) -> Complex<T> {
        self.eval(&Complex::new(T::zero(), s))
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use proptest::prelude::*;

    use std::str::FromStr;

    use super::*;
    use crate::{
        plots::{bode::Bode, polar::Polar},
        poly,
        polynomial::Poly,
        units::RadiansPerSecond,
    };

    #[test]
    fn delay() {
        let d = Tf::delay(Seconds(2.));
        assert_relative_eq!(1., d(Complex::new(0., 10.)).norm());
        assert_relative_eq!(-1., d(Complex::new(0., 0.5)).arg());
    }

    proptest! {
    #[test]
        fn qc_static_gain(g: f32) {
            let tf = Tf::new(poly!(g, -3.), poly!(1., 5., -0.5));
            assert_relative_eq!(g, tf.static_gain());
        }
    }

    #[test]
    fn stability() {
        let stable_den = Poly::new_from_roots(&[-1., -2.]);
        let stable_tf = Tf::new(poly!(1., 2.), stable_den);
        assert!(stable_tf.is_stable());

        let unstable_den = Poly::new_from_roots(&[0., -2.]);
        let unstable_tf = Tf::new(poly!(1., 2.), unstable_den);
        assert!(!unstable_tf.is_stable());
    }

    #[test]
    fn bode() {
        let tf = Tf::new(Poly::<f64>::one(), Poly::new_from_roots(&[-1.]));
        let b = Bode::new(tf, RadiansPerSecond(0.1), RadiansPerSecond(100.0), 0.1);
        for g in b.into_iter().into_db_deg() {
            assert!(g.magnitude() < 0.);
            assert!(g.phase() < 0.);
        }
    }

    #[test]
    fn polar() {
        let tf = Tf::new(poly!(5.), Poly::new_from_roots(&[-1., -10.]));
        let p = Polar::new(tf, RadiansPerSecond(0.1), RadiansPerSecond(10.0), 0.1);
        for g in p {
            assert!(g.magnitude() < 1.);
            assert!(g.phase() < 0.);
        }
    }

    #[test]
    fn initial_value() {
        let tf = Tf::new(poly!(4.), poly!(1., 5.));
        assert_relative_eq!(0., tf.init_value());
        let tf = Tf::new(poly!(4., -12.), poly!(1., 5.));
        assert_relative_eq!(-2.4, tf.init_value());
        let tf = Tf::new(poly!(-3., 4.), poly!(5.));
        assert_relative_eq!(std::f32::INFINITY, tf.init_value());
    }

    #[test]
    fn derivative_initial_value() {
        let tf = Tf::new(poly!(1., -3.), poly!(1., 3., 2.));
        assert_relative_eq!(-1.5, tf.init_value_der());
        let tf = Tf::new(poly!(1.), poly!(1., 3., 2.));
        assert_relative_eq!(0., tf.init_value_der());
        let tf = Tf::new(poly!(1., 0.5, -3.), poly!(1., 3., 2.));
        assert_relative_eq!(std::f32::INFINITY, tf.init_value_der());
    }

    #[test]
    fn complementary_sensitivity() {
        let g = Tf::new(poly!(1.), poly!(0., 1.));
        let r = Tf::new(poly!(4.), poly!(1., 1.));
        let f = g.compl_sensitivity(&r);
        assert_eq!(Tf::new(poly!(4.), poly!(4., 1., 1.)), f);
    }

    #[test]
    fn sensitivity() {
        let g = Tf::new(poly!(1.), poly!(0., 1.));
        let r = Tf::new(poly!(4.), poly!(1., 1.));
        let s = g.sensitivity(&r);
        assert_eq!(Tf::new(poly!(0., 1., 1.), poly!(4., 1., 1.)), s);
    }

    #[test]
    fn control_sensitivity() {
        let g = Tf::new(poly!(1.), poly!(0., 1.));
        let r = Tf::new(poly!(4.), poly!(1., 1.));
        let q = g.control_sensitivity(&r);
        assert_eq!(Tf::new(poly!(0., 4.), poly!(4., 1., 1.)), q);
    }

    #[test]
    fn root_locus() {
        let l = Tf::new(poly!(1.), Poly::new_from_roots(&[-1., -2.]));

        let locus1 = l.root_locus(0.25);
        assert_eq!(Complex::from_str("-1.5").unwrap(), locus1[0]);

        let locus2 = l.root_locus(-2.);
        assert_eq!(Complex::from_str("-3.").unwrap(), locus2[0]);
        assert_eq!(Complex::from_str("0.").unwrap(), locus2[1]);
    }

    #[test]
    fn root_locus_iterations() {
        let l = Tf::new(poly!(1.0_f32), Poly::new_from_roots(&[0., -3., -5.]));
        let loci = l.root_locus_plot(1., 130., 1.).into_iter();
        let last = loci.last().unwrap();
        assert_relative_eq!(130., last.k());
        assert_eq!(3, last.output().len());
        assert!(last.output().iter().any(|r| r.re > 0.));
    }
}
