//! # Transfer functions for discrete time systems.
//!
//! Specialized struct and methods for discrete time transfer functions
//! * time delay
//! * initial value
//! * static gain
//! * ARMA (autoregressive moving average) time evaluation method
//!
//! This module contains the discretization struct of a continuous time
//! transfer function
//! * forward Euler mehtod
//! * backward Euler method
//! * Tustin (trapezoidal) method

use num_complex::Complex;
use num_traits::{Float, MulAdd};

use std::{cmp::Ordering, collections::VecDeque, fmt::Debug, iter::Sum, ops::Mul};

use crate::{transfer_function::TfGen, Discrete, Eval};

/// Discrete transfer function
pub type Tfz<T> = TfGen<T, Discrete>;

impl<T: Float> Tfz<T> {
    /// Time delay for discrete time transfer function.
    /// `y(k) = u(k - h)`
    /// `G(z) = z^(-h)
    ///
    /// # Arguments
    ///
    /// * `h` - Time delay
    ///
    /// # Example
    /// ```
    /// use num_complex::Complex;
    /// use automatica::{units::Seconds, Tfz};
    /// let d = Tfz::delay(2);
    /// assert_eq!(0.010000001, d(Complex::new(0., 10.0_f32)).norm());
    /// ```
    pub fn delay(k: i32) -> impl Fn(Complex<T>) -> Complex<T> {
        move |z| z.powi(-k)
    }

    /// System inital value response to step input.
    /// `y(0) = G(z->infinity)`
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tf = Tfz::new(poly!(4.), poly!(1., 5.));
    /// assert_eq!(0., tf.init_value());
    /// ```
    #[must_use]
    pub fn init_value(&self) -> T {
        let n = self.num.degree();
        let d = self.den.degree();
        match n.cmp(&d) {
            Ordering::Less => T::zero(),
            Ordering::Equal => self.num.leading_coeff() / self.den.leading_coeff(),
            Ordering::Greater => T::infinity(),
        }
    }
}

impl<T: Float + MulAdd<Output = T>> Tfz<T> {
    /// Static gain `G(1)`.
    /// Ratio between constant output and constant input.
    /// Static gain is defined only for transfer functions of 0 type.
    ///
    /// Example
    ///
    /// ```
    /// use automatica::{poly, Tfz};
    /// let tf = Tfz::new(poly!(5., -3.),poly!(2., 5., -6.));
    /// assert_eq!(2., tf.static_gain());
    /// ```
    #[must_use]
    pub fn static_gain(&self) -> T {
        self.eval(T::one())
    }
}

/// Macro defining the common behaviour when creating the arma iterator.
///
/// # Arguments
///
/// * `self` - `self` parameter keyword
/// * `y_coeffs` - vector containing the coefficients of the output
/// * `u_coeffs` - vector containing the coefficients of the input
/// * `y` - queue containing the calculated outputs
/// * `u` - queue containing the supplied inputs
macro_rules! arma {
    ($self:ident, $y_coeffs:ident, $u_coeffs:ident, $y:ident, $u:ident) => {{
        let g = $self.normalize();
        let n_n = g.num.degree().unwrap_or(0);
        let n_d = g.den.degree().unwrap_or(0);
        let n = n_n.max(n_d);

        // The front is the lowest order coefficient.
        // The back is the higher order coefficient.
        // The higher degree terms are the more recent.
        // The last coefficient is always 1, because g is normalized.
        // [a0, a1, a2, ..., a(n-1), 1]
        let mut output_coefficients = g.den.coeffs();
        // Remove the last coefficient by truncating the vector by one.
        // This is done because the last coefficient of the denominator corresponds
        // to the currently calculated output.
        output_coefficients.truncate(n_d);
        // [a0, a1, a2, ..., a(n-1)]
        $y_coeffs = output_coefficients;
        // [b0, b1, b2, ..., bm]
        $u_coeffs = g.num.coeffs();

        // The coefficients do not need to be extended with zeros,
        // when the coffiecients are 'zipped' with the VecDeque, the zip stops at the
        // shortest iterator.

        let length = n + 1;
        // The front is the oldest calculated output.
        // [y(k-n), y(k-n+1), ..., y(k-1), y(k)]
        $y = VecDeque::from(vec![T::zero(); length]);
        // The front is the oldest input.
        // [u(k-n), u(k-n+1), ..., u(k-1), u(k)]
        $u = VecDeque::from(vec![T::zero(); length]);
    }};
}

impl<T: Float + Mul<Output = T> + Sum> Tfz<T> {
    /// Autoregressive moving average representation of a discrete transfer function
    /// It transforms the transfer function into time domain input-output
    /// difference equation.
    /// ```text
    ///                   b_n*z^n + b_(n-1)*z^(n-1) + ... + b_1*z + b_0
    /// Y(z) = G(z)U(z) = --------------------------------------------- U(z)
    ///                     z^n + a_(n-1)*z^(n-1) + ... + a_1*z + a_0
    ///
    /// y(k) = - a_(n-1)*y(k-1) - ... - a_1*y(k-n+1) - a_0*y(k-n) +
    ///        + b_n*u(k) + b_(n-1)*u(k-1) + ... + b_1*u(k-n+1) + b_0*u(k-n)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - Input function
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, signals::discrete, Tfz};
    /// let tfz = Tfz::new(poly!(1., 2., 3.), poly!(0., 0., 0., 1.));
    /// let mut iter = tfz.arma_fn(discrete::step(1., 0));
    /// assert_eq!(Some(0.), iter.next());
    /// assert_eq!(Some(3.), iter.next());
    /// assert_eq!(Some(5.), iter.next());
    /// assert_eq!(Some(6.), iter.next());
    /// ```
    pub fn arma_fn<F>(&self, input: F) -> ArmaFn<F, T>
    where
        F: Fn(usize) -> T,
    {
        let y_coeffs: Vec<_>;
        let u_coeffs: Vec<_>;
        let y: VecDeque<_>;
        let u: VecDeque<_>;
        arma!(self, y_coeffs, u_coeffs, y, u);

        ArmaFn {
            y_coeffs,
            u_coeffs,
            y,
            u,
            input,
            k: 0,
        }
    }

    /// Autoregressive moving average representation of a discrete transfer function
    /// It transforms the transfer function into time domain input-output
    /// difference equation.
    /// ```text
    ///                   b_n*z^n + b_(n-1)*z^(n-1) + ... + b_1*z + b_0
    /// Y(z) = G(z)U(z) = --------------------------------------------- U(z)
    ///                     z^n + a_(n-1)*z^(n-1) + ... + a_1*z + a_0
    ///
    /// y(k) = - a_(n-1)*y(k-1) - ... - a_1*y(k-n+1) - a_0*y(k-n) +
    ///        + b_n*u(k) + b_(n-1)*u(k-1) + ... + b_1*u(k-n+1) + b_0*u(k-n)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator supplying the input data to the model
    ///
    /// # Example
    /// ```
    /// use automatica::{poly, signals::discrete, Tfz};
    /// let tfz = Tfz::new(poly!(1., 2., 3.), poly!(0., 0., 0., 1.));
    /// let mut iter = tfz.arma_iter(std::iter::repeat(1.));
    /// assert_eq!(Some(0.), iter.next());
    /// assert_eq!(Some(3.), iter.next());
    /// assert_eq!(Some(5.), iter.next());
    /// assert_eq!(Some(6.), iter.next());
    /// ```
    pub fn arma_iter<I, II>(&self, iter: II) -> ArmaIter<I, T>
    where
        II: IntoIterator<Item = T, IntoIter = I>,
        I: Iterator<Item = T>,
    {
        let y_coeffs: Vec<_>;
        let u_coeffs: Vec<_>;
        let y: VecDeque<_>;
        let u: VecDeque<_>;
        arma!(self, y_coeffs, u_coeffs, y, u);

        ArmaIter {
            y_coeffs,
            u_coeffs,
            y,
            u,
            iter: iter.into_iter(),
        }
    }
}

/// Iterator for the autoregressive moving average model of a discrete
/// transfer function.
/// The input is supplied through a function.
#[derive(Debug)]
pub struct ArmaFn<F, T>
where
    F: Fn(usize) -> T,
{
    /// y coefficients
    y_coeffs: Vec<T>,
    /// u coefficients
    u_coeffs: Vec<T>,
    /// y queue buffer
    y: VecDeque<T>,
    /// u queue buffer
    u: VecDeque<T>,
    /// input function
    input: F,
    /// step
    k: usize,
}

/// Macro containing the common iteration steps of the ARMA model
///
/// # Arguments
///
/// * `self` - `self` keyword parameter
macro_rules! arma_iter {
    ($self:ident, $current_input:ident) => {{
        // Push the current input into the most recent position of the input buffer.
        $self.u.push_back($current_input);
        // Discard oldest input.
        $self.u.pop_front();
        let input: T = $self
            .u_coeffs
            .iter()
            .zip(&$self.u)
            .map(|(&c, &u)| c * u)
            .sum();

        // Push zero in the last position shifting output values one step back
        // in time, zero suppress last coefficient which shall be the current
        // calculated output value.
        $self.y.push_back(T::zero());
        // Discard oldest output.
        $self.y.pop_front();
        let old_output: T = $self
            .y_coeffs
            .iter()
            .zip(&$self.y)
            .map(|(&c, &y)| c * y)
            .sum();

        // Calculate the output.
        let new_y = input - old_output;
        // Put the new calculated value in the last position of the buffer.
        // `back_mut` returns None if the Deque is empty, this should never happen.
        debug_assert!(!$self.y.is_empty());
        *$self.y.back_mut()? = new_y;
        Some(new_y)
    }};
}

impl<F, T> Iterator for ArmaFn<F, T>
where
    F: Fn(usize) -> T,
    T: Float + Mul<Output = T> + Sum,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_input = (self.input)(self.k);
        self.k += 1;
        arma_iter!(self, current_input)
    }
}

/// Iterator for the autoregressive moving average model of a discrete
/// transfer function.
/// The input is supplied through an iterator.
#[derive(Debug)]
pub struct ArmaIter<I, T>
where
    I: Iterator,
{
    /// y coefficients
    y_coeffs: Vec<T>,
    /// u coefficients
    u_coeffs: Vec<T>,
    /// y queue buffer
    y: VecDeque<T>,
    /// u queue buffer
    u: VecDeque<T>,
    /// input iterator
    iter: I,
}

impl<I, T> Iterator for ArmaIter<I, T>
where
    I: Iterator<Item = T>,
    T: Float + Mul<Output = T> + Sum,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let current_input = self.iter.next()?;
        arma_iter!(self, current_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly, polynomial::Poly, signals::discrete, units::Decibel, Eval};
    use num_complex::Complex64;

    #[test]
    fn tfz() {
        let _ = Tfz::new(poly!(1.), poly!(1., 2., 3.));
    }

    #[test]
    fn delay() {
        let d = Tfz::delay(2);
        assert_relative_eq!(0.010_000_001, d(Complex::new(0., 10.0_f32)).norm());
    }

    #[test]
    fn initial_value() {
        let tf = Tfz::new(poly!(4.), poly!(1., 5.));
        assert_relative_eq!(0., tf.init_value());

        let tf = Tfz::new(poly!(4., 10.), poly!(1., 5.));
        assert_relative_eq!(2., tf.init_value());

        let tf = Tfz::new(poly!(4., 1.), poly!(5.));
        assert_relative_eq!(std::f32::INFINITY, tf.init_value());
    }

    #[test]
    fn static_gain() {
        let tf = Tfz::new(poly!(5., -3.), poly!(2., 5., -6.));
        assert_relative_eq!(2., tf.static_gain());
    }

    #[test]
    fn eval() {
        let tf = Tfz::new(
            Poly::new_from_coeffs(&[2., 20.]),
            Poly::new_from_coeffs(&[1., 0.1]),
        );
        let z = 0.5 * Complex64::i();
        let g = tf.eval(z);
        assert_relative_eq!(20.159, g.norm().to_db(), max_relative = 1e-4);
        assert_relative_eq!(75.828, g.arg().to_degrees(), max_relative = 1e-4);
    }

    #[test]
    fn arma() {
        let tfz = Tfz::new(poly!(0.5_f32), poly!(-0.5, 1.));
        let mut iter = tfz.arma_fn(discrete::impulse(1., 0)).take(6);
        assert_eq!(Some(0.), iter.next());
        assert_eq!(Some(0.5), iter.next());
        assert_eq!(Some(0.25), iter.next());
        assert_eq!(Some(0.125), iter.next());
        assert_eq!(Some(0.0625), iter.next());
        assert_eq!(Some(0.03125), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn arma_iter() {
        use std::iter;
        let tfz = Tfz::new(poly!(0.5_f32), poly!(-0.5, 1.));
        let mut iter = tfz.arma_iter(iter::once(1.).chain(iter::repeat(0.)).take(6));
        assert_eq!(Some(0.), iter.next());
        assert_eq!(Some(0.5), iter.next());
        assert_eq!(Some(0.25), iter.next());
        assert_eq!(Some(0.125), iter.next());
        assert_eq!(Some(0.0625), iter.next());
        assert_eq!(Some(0.03125), iter.next());
        assert_eq!(None, iter.next());
    }
}
