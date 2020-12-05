//! # Ordinary differential equations solvers
//!
//! `Rk2` is an explicit Runge-Kutta of order 2 with 2 steps, it is suitable for
//! non stiff systems.
//!
//! `Rk4` is an explicit Runge-Kutta of order 4 with 4 steps, it is suitable for
//! non stiff systems.
//!
//! `Rkf45` is an explicit Runge-Kutta-Fehlberg of order 4 and 5 with 6 steps
//! and adaptive integration step, it is suitable for non stiff systems.
//!
//! `Radau` is an implicit Runge-Kutta-Radau of order 3 with 2 steps, it is
//! suitable for stiff systems.

use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{ComplexField, DMatrix, DVector, Dynamic, Scalar, SimdPartialOrd, LU};
use num_traits::{Float, Signed};

use std::{
    marker::Sized,
    ops::{AddAssign, MulAssign, SubAssign},
};

use crate::{linear_system::continuous::Ss, units::Seconds};

/// Define the order of the Runge-Kutta method.
#[derive(Clone, Debug)]
pub(super) enum Order {
    /// Runge-Kutta method of order 2.
    Rk2,
    /// Runge-Kutta method of order 4.
    Rk4,
}

/// Struct for the time evolution of a linear system
#[derive(Clone, Debug)]
pub struct Rk<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: Float + Scalar,
{
    /// Linear system
    sys: &'a Ss<T>,
    /// Input function
    input: F,
    /// State vector.
    state: DVector<T>,
    /// Output vector.
    output: DVector<T>,
    /// Interval.
    h: Seconds<T>,
    /// Number of steps.
    n: usize,
    /// Index.
    index: usize,
    /// Order of the solver.
    order: Order,
}

impl<'a, F, T> Rk<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AddAssign + Float + MulAssign + RkConst + Scalar,
{
    /// Create the solver for a Runge-Kutta method.
    ///
    /// # Arguments
    ///
    /// * `sys` - linear system
    /// * `u` - input function that returns a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    /// * `order` - order of the solver
    pub(super) fn new(
        sys: &'a Ss<T>,
        u: F,
        x0: &[T],
        h: Seconds<T>,
        n: usize,
        order: Order,
    ) -> Self {
        let start = DVector::from_vec(u(Seconds(T::zero())));
        let state = DVector::from_column_slice(x0);
        let output = &sys.c * &state + &sys.d * &start;
        Self {
            sys,
            input: u,
            state,
            output,
            h,
            n,
            index: 0,
            order,
        }
    }

    /// Initial step (time 0) of the Runge-Kutta solver.
    /// It contains the initial state and the calculated initial output
    /// at the constructor.
    fn initial_step(&mut self) -> Option<Step<T>> {
        self.index += 1;
        // State and output at time 0.
        Some(Step {
            time: Seconds(T::zero()),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Runge-Kutta order 2 method.
    #[allow(clippy::cast_precision_loss)]
    fn main_iteration_rk2(&mut self) -> Option<Step<T>> {
        // y_n+1 = y_n + 1/2(k1 + k2) + O(h^3)
        // k1 = h*f(t_n, y_n)
        // k2 = h*f(t_n + h, y_n + k1)
        // Retrun None if conversion fails.
        let init_time = Seconds(T::from(self.index - 1)? * self.h.0);
        let end_time = Seconds(T::from(self.index)? * self.h.0);
        let u = DVector::from_vec((self.input)(init_time));
        let uh = DVector::from_vec((self.input)(end_time));
        let bu = &self.sys.b * &u;
        let buh = &self.sys.b * &uh;
        let k1 = (&self.sys.a * &self.state + &bu) * self.h.0;
        let k2 = (&self.sys.a * (&self.state + &k1) + &buh) * self.h.0;
        self.state += (k1 + k2) * T::_05;
        self.output = &self.sys.c * &self.state + &self.sys.d * &uh;

        self.index += 1;
        Some(Step {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Runge-Kutta order 4 method.
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    fn main_iteration_rk4(&mut self) -> Option<Step<T>> {
        // y_n+1 = y_n + h/6(k1 + 2*k2 + 2*k3 + k4) + O(h^4)
        // k1 = f(t_n, y_n)
        // k2 = f(t_n + h/2, y_n + h/2 * k1)
        // k3 = f(t_n + h/2, y_n + h/2 * k2)
        // k2 = f(t_n + h, y_n + h*k3)
        // Return None if conversion fails
        let init_time = Seconds(T::from(self.index - 1)? * self.h.0);
        let mid_time = Seconds(init_time.0 + T::_05 * self.h.0);
        let end_time = Seconds(T::from(self.index)? * self.h.0);
        let u = DVector::from_vec((self.input)(init_time));
        let u_mid = DVector::from_vec((self.input)(mid_time));
        let u_end = DVector::from_vec((self.input)(end_time));
        let bu = &self.sys.b * &u;
        let bu_mid = &self.sys.b * &u_mid;
        let bu_end = &self.sys.b * &u_end;
        let k1 = &self.sys.a * &self.state + &bu;
        let k2 = &self.sys.a * (&self.state + &k1 * (T::_05 * self.h.0)) + &bu_mid;
        let k3 = &self.sys.a * (&self.state + &k2 * (T::_05 * self.h.0)) + &bu_mid;
        let k4 = &self.sys.a * (&self.state + &k3 * self.h.0) + &bu_end;
        let n_2 = T::A_RK[0];
        let n_6 = T::A_RK[1];
        self.state += (k1 + k2 * n_2 + k3 * n_2 + k4) * (self.h.0 / n_6);
        self.output = &self.sys.c * &self.state + &self.sys.d * &u_end;

        self.index += 1;
        Some(Step {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

// Coefficients of the Butcher table of rk method.
/// Trait that defines the constants used in the Rk solver.
pub trait RkConst
where
    Self: Copy + Sized,
{
    /// 0.5 constant
    const _05: Self;
    /// A
    const A_RK: [Self; 2];
}

macro_rules! impl_rk_const {
    ($t:ty) => {
        impl RkConst for $t {
            const _05: Self = 0.5;
            const A_RK: [Self; 2] = [2., 6.];
        }
    };
}

impl_rk_const!(f32);
impl_rk_const!(f64);
//////

/// Implementation of the Iterator trait for the `Rk` struct
impl<'a, F, T> Iterator for Rk<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AddAssign + Float + MulAssign + RkConst + Scalar,
{
    type Item = Step<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.n {
            None
        } else if self.index == 0 {
            self.initial_step()
        } else {
            match self.order {
                Order::Rk2 => self.main_iteration_rk2(),
                Order::Rk4 => self.main_iteration_rk4(),
            }
        }
    }
}

/// Struct to hold the data of the linear system time evolution
#[derive(Clone, Debug)]
pub struct Step<T: Float> {
    /// Time of the current step
    time: Seconds<T>,
    /// Current state
    state: Vec<T>,
    /// Current output
    output: Vec<T>,
}

impl<T: Float> Step<T> {
    /// Get the time of the current step
    pub fn time(&self) -> Seconds<T> {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<T> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }
}

/// Struct for the time evolution of a linear system
#[derive(Clone, Debug)]
pub struct Rkf45<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: Float + Scalar,
{
    /// Linear system
    sys: &'a Ss<T>,
    /// Input function
    input: F,
    /// State vector.
    state: DVector<T>,
    /// Output vector.
    output: DVector<T>,
    /// Interval.
    h: Seconds<T>,
    /// Time limit of the evaluation
    limit: Seconds<T>,
    /// Time
    time: Seconds<T>,
    /// Tolerance
    tol: T,
    /// Is initial step
    initial_step: bool,
}

impl<'a, F, T> Rkf45<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AddAssign + Float + MulAssign + Rkf45Const + Scalar + Signed + SimdPartialOrd + SubAssign,
{
    /// Create a solver using Runge-Kutta-Fehlberg method
    ///
    /// # Arguments
    ///
    /// * `sys` - linear system
    /// * `u` - input function (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `limit` - time limit of the evaluation
    /// * `tol` - error tolerance
    pub(super) fn new(
        sys: &'a Ss<T>,
        u: F,
        x0: &[T],
        h: Seconds<T>,
        limit: Seconds<T>,
        tol: T,
    ) -> Self {
        let start = DVector::from_vec(u(Seconds(T::zero())));
        let state = DVector::from_column_slice(x0);
        // Calculate the output at time 0.
        let output = &sys.c * &state + &sys.d * &start;
        Self {
            sys,
            input: u,
            state,
            output,
            h,
            limit,
            time: Seconds(T::zero()),
            tol,
            initial_step: true,
        }
    }

    /// Initial step (time 0) of the rkf45 solver.
    /// It contains the initial state and the calculated initial output
    /// at the constructor
    fn initial_step(&mut self) -> Option<StepWithError<T>> {
        self.initial_step = false;
        Some(StepWithError {
            time: Seconds(T::zero()),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
            error: T::zero(),
        })
    }

    /// Runge-Kutta-Fehlberg order 4 and 5 method with adaptive step size
    fn main_iteration(&mut self) -> Option<StepWithError<T>> {
        let mut error;
        loop {
            let u1 = DVector::from_vec((self.input)(self.time));
            let u2 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * T::A[0])));
            let u3 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * T::A[1])));
            let u4 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * T::A[2])));
            let u5 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0)));
            let u6 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * T::A[3])));

            let k1 = (&self.sys.a * &self.state + &self.sys.b * &u1) * self.h.0;
            let k2 = (&self.sys.a * (&self.state + &k1 * T::B21) + &self.sys.b * &u2) * self.h.0;
            let k3 = (&self.sys.a * (&self.state + &k1 * T::B3[0] + &k2 * T::B3[1])
                + &self.sys.b * &u3)
                * self.h.0;
            let k4 = (&self.sys.a
                * (&self.state + &k1 * T::B4[0] + &k2 * T::B4[1] + &k3 * T::B4[2])
                + &self.sys.b * &u4)
                * self.h.0;
            let k5 = (&self.sys.a
                * (&self.state
                    + &k1 * T::B5[0]
                    + &k2 * T::B5[1]
                    + &k3 * T::B5[2]
                    + &k4 * T::B5[3])
                + &self.sys.b * &u5)
                * self.h.0;
            let k6 = (&self.sys.a
                * (&self.state
                    + &k1 * T::B6[0]
                    + &k2 * T::B6[1]
                    + &k3 * T::B6[2]
                    + &k4 * T::B6[3]
                    + &k5 * T::B6[4])
                + &self.sys.b * &u6)
                * self.h.0;

            let xn1 = &self.state + &k1 * T::C[0] + &k3 * T::C[1] + &k4 * T::C[2] + &k5 * T::C[3];
            let xn1_ = &self.state
                + &k1 * T::D[0]
                + &k3 * T::D[1]
                + &k4 * T::D[2]
                + &k5 * T::D[3]
                + &k6 * T::D[4];

            // Take the maximum absolute error between the states of the system.
            error = (&xn1 - &xn1_).abs().max();
            let error_ratio = self.tol / error;
            if error < self.tol {
                self.h.0 = T::SAFETY_FACTOR * self.h.0 * error_ratio.powf(T::EXP[0]);
                self.state = xn1;
                break;
            }
            self.h.0 = T::SAFETY_FACTOR * self.h.0 * error_ratio.powf(T::EXP[1]);
        }

        // Update time before calculate the output.
        self.time.0 += self.h.0;

        let u = DVector::from_vec((self.input)(self.time));
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        Some(StepWithError {
            time: self.time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
            error,
        })
    }
}

/// Implementation of the Iterator trait for the `Rkf45` struct
impl<'a, F, T> Iterator for Rkf45<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AddAssign + Float + MulAssign + Rkf45Const + Signed + Scalar + SimdPartialOrd + SubAssign,
{
    type Item = StepWithError<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.time > self.limit {
            None
        } else if self.initial_step {
            self.initial_step()
        } else {
            self.main_iteration()
        }
    }
}

// Coefficients of the Butcher table of rkf45 method.
/// Trait that defines the constants used in the Rkf45 solver.
pub trait Rkf45Const
where
    Self: Copy + Sized,
{
    /// A
    const A: [Self; 4];
    /// B21
    const B21: Self;
    /// B3
    const B3: [Self; 2];
    /// B4
    const B4: [Self; 3];
    /// B5
    const B5: [Self; 4];
    /// B6
    const B6: [Self; 5];
    /// C
    const C: [Self; 4];
    /// D
    const D: [Self; 5];
    /// Safety factor to avoid too small step changes.
    const SAFETY_FACTOR: Self;
    /// Error ratio exponents.
    const EXP: [Self; 2];
}

macro_rules! impl_rkf45_const {
    ($t:ty) => {
        impl Rkf45Const for $t {
            const A: [Self; 4] = [1. / 4., 3. / 8., 12. / 13., 1. / 2.];
            const B21: Self = 1. / 4.;
            const B3: [Self; 2] = [3. / 32., 9. / 32.];
            const B4: [Self; 3] = [1932. / 2197., -7200. / 2197., 7296. / 2197.];
            const B5: [Self; 4] = [439. / 216., -8., 3680. / 513., -845. / 4104.];
            const B6: [Self; 5] = [-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.];
            const C: [Self; 4] = [25. / 216., 1408. / 2564., 2197. / 4101., -1. / 5.];
            const D: [Self; 5] = [
                16. / 135.,
                6656. / 12_825.,
                28_561. / 56_430.,
                -9. / 50.,
                2. / 55.,
            ];
            const SAFETY_FACTOR: Self = 0.95;
            const EXP: [Self; 2] = [0.25, 0.2];
        }
    };
}

impl_rkf45_const!(f32);
impl_rkf45_const!(f64);
//////

/// Struct to hold the data of the linear system time evolution
#[derive(Clone, Debug)]
pub struct StepWithError<T: Float> {
    /// Current step size
    time: Seconds<T>,
    /// Current state
    state: Vec<T>,
    /// Current output
    output: Vec<T>,
    /// Current maximum absolute error
    error: T,
}

impl<T: Float> StepWithError<T> {
    /// Get the time of the current step
    pub fn time(&self) -> Seconds<T> {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<T> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<T> {
        &self.output
    }

    /// Get the current maximum absolute error
    pub fn error(&self) -> T {
        self.error
    }
}

/// Struct for the time evolution of the linear system using the implicit
/// Radau method of order 3 with 2 steps
#[derive(Clone, Debug)]
pub struct Radau<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: ComplexField + Float + Scalar,
{
    /// Linear system
    sys: &'a Ss<T>,
    /// Input function
    input: F,
    /// State vector
    state: DVector<T>,
    /// Output vector
    output: DVector<T>,
    /// Interval
    h: Seconds<T>,
    /// Number of steps
    n: usize,
    /// Index
    index: usize,
    /// Tolerance
    tol: T,
    /// Store the LU decomposition of the Jacobian matrix
    lu_jacobian: LU<T, Dynamic, Dynamic>,
}

impl<'a, F, T> Radau<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AbsDiffEq<Epsilon = T> + ComplexField + Float + Scalar + RadauConst + RelativeEq,
{
    /// Create the solver for a Radau order 3 with 2 steps method.
    ///
    /// # Arguments
    ///
    /// * `sys` - linear system
    /// * `u` - input function that returns a vector (column vector)
    /// * `x0` - initial state (column vector)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    /// * `tol` - tolerance of implicit solution finding
    pub(super) fn new(sys: &'a Ss<T>, u: F, x0: &[T], h: Seconds<T>, n: usize, tol: T) -> Self {
        let start = DVector::from_vec(u(Seconds(T::zero())));
        let state = DVector::from_column_slice(x0);
        let output = &sys.c * &state + &sys.d * &start;
        // Jacobian matrix can be precomputed since it is constant for the
        // given system.
        let g = &sys.a * h.0;
        let rows = &sys.a.nrows(); // A is a square matrix.
        let identity = DMatrix::<T>::identity(*rows, *rows);
        let j11 = &g * T::RADAU_A[0] - &identity;
        let j12 = &g * T::RADAU_A[1];
        let j21 = &g * T::RADAU_A[2];
        let j22 = &g * T::RADAU_A[3] - &identity;
        let mut jac = DMatrix::zeros(2 * *rows, 2 * *rows);
        // Copy the sub matrices into the Jacobian.
        let sub_matrix_size = (*rows, *rows);
        jac.slice_mut((0, 0), sub_matrix_size).copy_from(&j11);
        jac.slice_mut((0, *rows), sub_matrix_size).copy_from(&j12);
        jac.slice_mut((*rows, 0), sub_matrix_size).copy_from(&j21);
        jac.slice_mut((*rows, *rows), sub_matrix_size)
            .copy_from(&j22);

        Self {
            sys,
            input: u,
            state,
            output,
            h,
            n,
            index: 0,
            tol,
            lu_jacobian: jac.lu(),
        }
    }

    /// Initial step (time 0) of the Radau solver.
    /// It contains the initial state and the calculated initial output
    /// at the constructor.
    fn initial_step(&mut self) -> Option<Step<T>> {
        self.index += 1;
        Some(Step {
            time: Seconds(T::zero()),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Radau order 3 with 2 step implicit method.
    #[allow(clippy::cast_precision_loss)]
    fn main_iteration(&mut self) -> Option<Step<T>> {
        // Return None if conversion fails.
        let time = T::from(self.index - 1)? * self.h.0;
        let rows = self.sys.a.nrows();
        // k = [k1; k2] (column vector)
        let mut k = DVector::<T>::zeros(2 * rows);
        // k sub-vectors (or block vectors) are have size (rows x 1).
        let sub_vec_size = (rows, 1);
        // Use as first guess for k1 and k2 the current state.
        k.slice_mut((0, 0), sub_vec_size).copy_from(&self.state);
        k.slice_mut((rows, 0), sub_vec_size).copy_from(&self.state);

        let u1 = DVector::from_vec((self.input)(Seconds(time + T::RADAU_C[0] * self.h.0)));
        let bu1 = &self.sys.b * &u1;
        let u2 = DVector::from_vec((self.input)(Seconds(time + T::RADAU_C[1] * self.h.0)));
        let bu2 = &self.sys.b * &u2;
        let mut f = DVector::<T>::zeros(2 * rows);
        // Max 10 iterations.
        for _ in 0..10 {
            let k1 = k.slice((0, 0), sub_vec_size);
            let k2 = k.slice((rows, 0), sub_vec_size);

            let f1 = &self.sys.a
                * (&self.state + (k1 * T::RADAU_A[0] + k2 * T::RADAU_A[1]) * self.h.0)
                + &bu1
                - k1;
            let f2 = &self.sys.a
                * (&self.state + (k1 * T::RADAU_A[2] + k2 * T::RADAU_A[3]) * self.h.0)
                + &bu2
                - k2;
            f.slice_mut((0, 0), sub_vec_size).copy_from(&f1);
            f.slice_mut((rows, 0), sub_vec_size).copy_from(&f2);

            // J * dk = f -> dk = J^-1 * f
            // Override f with dk so there is less allocations of matrices.
            // f = J^-1 * f
            let knew = if self.lu_jacobian.solve_mut(&mut f) {
                // k(n+1) = k(n) - dk = k(n) - f
                &k - &f
            } else {
                eprintln!("Unable to solve step {} at time {}", self.index, time);
                return None;
            };

            let eq = &knew.relative_eq(&k, self.tol, T::RADAU_ABS_TOL);
            k = knew; // Use the latest solution calculated.
            if *eq {
                break;
            }
        }
        self.state += (k.slice((0, 0), (rows, 1)) * T::RADAU_B[0]
            + k.slice((rows, 0), (rows, 1)) * T::RADAU_B[1])
            * self.h.0;

        // Return None if conversion fails.
        let end_time = Seconds(T::from(self.index)? * self.h.0);
        let u = DVector::from_vec((self.input)(end_time));
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        self.index += 1;
        Some(Step {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

// Constants for Radau method.
/// Trait that defines the constants used in the Radau solver.
pub trait RadauConst
where
    Self: Copy + Sized,
{
    /// A
    const RADAU_A: [Self; 4];
    /// B
    const RADAU_B: [Self; 2];
    /// C
    const RADAU_C: [Self; 2];
    /// Absolute tolerance for equality comparisons.
    const RADAU_ABS_TOL: Self;
}

macro_rules! impl_radau_const {
    ($t:ty) => {
        impl RadauConst for $t {
            const RADAU_A: [Self; 4] = [5. / 12., -1. / 12., 3. / 4., 1. / 4.];
            const RADAU_B: [Self; 2] = [3. / 4., 1. / 4.];
            const RADAU_C: [Self; 2] = [1. / 3., 1.];
            const RADAU_ABS_TOL: Self = 0.001;
        }
    };
}

impl_radau_const!(f32);
impl_radau_const!(f64);
//////

/// Implementation of the Iterator trait for the `Radau` struct.
impl<'a, F, T> Iterator for Radau<'a, F, T>
where
    F: Fn(Seconds<T>) -> Vec<T>,
    T: AbsDiffEq<Epsilon = T> + ComplexField + Float + Scalar + RadauConst + RelativeEq,
{
    type Item = Step<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.n {
            None
        } else if self.index == 0 {
            self.initial_step()
        } else {
            self.main_iteration()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runge_kutta_struct() {
        let t = Seconds(3.);
        let s = vec![2., 3.];
        let o = vec![-5., -4.];

        let rk = Step {
            time: t,
            state: s.clone(),
            output: o.clone(),
        };
        assert_eq!(t, rk.time());
        assert_eq!(&s, rk.state());
        assert_eq!(&o, rk.output());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn runge_kutta_fehlberg_struct() {
        let t = Seconds(3.);
        let s = vec![2., 3.];
        let o = vec![-5., -4.];
        let e = 0.5;

        let rkf = StepWithError {
            time: t,
            state: s.clone(),
            output: o.clone(),
            error: e,
        };
        assert_eq!(t, rkf.time());
        assert_eq!(&s, rkf.state());
        assert_eq!(&o, rkf.output());
        assert_eq!(e, rkf.error());
    }

    #[test]
    fn radau_struct() {
        let t = Seconds(12.);
        let s = vec![2., 2.4];
        let o = vec![-5.33, -4.];

        let rd = Step {
            time: t,
            state: s.clone(),
            output: o.clone(),
        };
        assert_eq!(t, rd.time());
        assert_eq!(&s, rd.state());
        assert_eq!(&o, rd.output());
    }
}
