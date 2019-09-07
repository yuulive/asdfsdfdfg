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

use crate::{linear_system::Ss, Seconds};

use nalgebra::{DMatrix, DVector, Dynamic, LU};

/// Define the order of the Runge-Kutta method.
#[derive(Debug)]
pub(crate) enum Order {
    /// Runge-Kutta method of order 2.
    Rk2,
    /// Runge-Kutta method of order 4.
    Rk4,
}

/// Struct for the time evolution of a linear system
#[derive(Debug)]
pub struct RkIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    /// Linear system
    sys: &'a Ss,
    /// Input function
    input: F,
    /// State vector.
    state: DVector<f64>,
    /// Output vector.
    output: DVector<f64>,
    /// Interval.
    h: Seconds,
    /// Number of steps.
    n: usize,
    /// Index.
    index: usize,
    /// Order of the solver.
    order: Order,
}

impl<'a, F> RkIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
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
    pub(crate) fn new(sys: &'a Ss, u: F, x0: &[f64], h: Seconds, n: usize, order: Order) -> Self {
        let start = DVector::from_vec(u(Seconds(0.)));
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
    fn initial_step(&mut self) -> Option<Rk> {
        self.index += 1;
        // State and output at time 0.
        Some(Rk {
            time: Seconds(0.),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Runge-Kutta order 2 method.
    #[allow(clippy::cast_precision_loss)]
    fn main_iteration_rk2(&mut self) -> Option<Rk> {
        // y_n+1 = y_n + 1/2(k1 + k2) + O(h^3)
        // k1 = h*f(t_n, y_n)
        // k2 = h*f(t_n + h, y_n + k1)
        let init_time = Seconds((self.index - 1) as f64 * self.h.0);
        let end_time = Seconds(self.index as f64 * self.h.0);
        let u = DVector::from_vec((self.input)(init_time));
        let uh = DVector::from_vec((self.input)(end_time));
        let bu = &self.sys.b * &u;
        let buh = &self.sys.b * &uh;
        let k1 = self.h.0 * (&self.sys.a * &self.state + &bu);
        let k2 = self.h.0 * (&self.sys.a * (&self.state + &k1) + &buh);
        self.state += 0.5 * (k1 + k2);
        self.output = &self.sys.c * &self.state + &self.sys.d * &uh;

        self.index += 1;
        Some(Rk {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Runge-Kutta order 4 method.
    #[allow(clippy::cast_precision_loss)]
    fn main_iteration_rk4(&mut self) -> Option<Rk> {
        // y_n+1 = y_n + h/6(k1 + 2*k2 + 2*k3 + k4) + O(h^4)
        // k1 = f(t_n, y_n)
        // k2 = f(t_n + h/2, y_n + h/2 * k1)
        // k3 = f(t_n + h/2, y_n + h/2 * k2)
        // k2 = f(t_n + h, y_n + h*k3)
        let init_time = Seconds((self.index - 1) as f64 * self.h.0);
        let mid_time = Seconds(init_time.0 + 0.5 * self.h.0);
        let end_time = Seconds(self.index as f64 * self.h.0);
        let u = DVector::from_vec((self.input)(init_time));
        let u_mid = DVector::from_vec((self.input)(mid_time));
        let u_end = DVector::from_vec((self.input)(end_time));
        let bu = &self.sys.b * &u;
        let bu_mid = &self.sys.b * &u_mid;
        let bu_end = &self.sys.b * &u_end;
        let k1 = &self.sys.a * &self.state + &bu;
        let k2 = &self.sys.a * (&self.state + 0.5 * self.h.0 * &k1) + &bu_mid;
        let k3 = &self.sys.a * (&self.state + 0.5 * self.h.0 * &k2) + &bu_mid;
        let k4 = &self.sys.a * (&self.state + self.h.0 * &k3) + &bu_end;
        self.state += self.h.0 / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
        self.output = &self.sys.c * &self.state + &self.sys.d * &u_end;

        self.index += 1;
        Some(Rk {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

/// Implementation of the Iterator trait for the `RkIterator` struct
impl<'a, F> Iterator for RkIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    type Item = Rk;

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
#[derive(Debug)]
pub struct Rk {
    /// Time of the current step
    time: Seconds,
    /// Current state
    state: Vec<f64>,
    /// Current output
    output: Vec<f64>,
}

impl Rk {
    /// Get the time of the current step
    pub fn time(&self) -> Seconds {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<f64> {
        &self.output
    }
}

/// Struct for the time evolution of a linear system
#[derive(Debug)]
pub struct Rkf45Iterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    /// Linear system
    sys: &'a Ss,
    /// Input function
    input: F,
    /// State vector.
    state: DVector<f64>,
    /// Output vector.
    output: DVector<f64>,
    /// Interval.
    h: Seconds,
    /// Time limit of the evaluation
    limit: Seconds,
    /// Time
    time: Seconds,
    /// Tolerance
    tol: f64,
    /// Is initial step
    initial_step: bool,
}

impl<'a, F> Rkf45Iterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
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
    pub(crate) fn new(sys: &'a Ss, u: F, x0: &[f64], h: Seconds, limit: Seconds, tol: f64) -> Self {
        let start = DVector::from_vec(u(Seconds(0.)));
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
            time: Seconds(0.),
            tol,
            initial_step: true,
        }
    }

    /// Initial step (time 0) of the rkf45 solver.
    /// It contains the initial state and the calculated initial output
    /// at the constructor
    fn initial_step(&mut self) -> Option<Rkf45> {
        self.initial_step = false;
        Some(Rkf45 {
            time: Seconds(0.),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
            error: 0.,
        })
    }

    /// Runge-Kutta-Fehlberg order 4 and 5 method with adaptive step size
    fn main_iteration(&mut self) -> Option<Rkf45> {
        let mut error;
        loop {
            let u1 = DVector::from_vec((self.input)(self.time));
            let u2 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * A[0])));
            let u3 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * A[1])));
            let u4 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * A[2])));
            let u5 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0)));
            let u6 = DVector::from_vec((self.input)(Seconds(self.time.0 + self.h.0 * A[3])));

            let k1 = self.h.0 * (&self.sys.a * &self.state + &self.sys.b * &u1);
            let k2 = self.h.0 * (&self.sys.a * (&self.state + B21 * &k1) + &self.sys.b * &u2);
            let k3 = self.h.0
                * (&self.sys.a * (&self.state + B3[0] * &k1 + B3[1] * &k2) + &self.sys.b * &u3);
            let k4 = self.h.0
                * (&self.sys.a * (&self.state + B4[0] * &k1 + B4[1] * &k2 + B4[2] * &k3)
                    + &self.sys.b * &u4);
            let k5 = self.h.0
                * (&self.sys.a
                    * (&self.state + B5[0] * &k1 + B5[1] * &k2 + B5[2] * &k3 + B5[3] * &k4)
                    + &self.sys.b * &u5);
            let k6 = self.h.0
                * (&self.sys.a
                    * (&self.state
                        + B6[0] * &k1
                        + B6[1] * &k2
                        + B6[2] * &k3
                        + B6[3] * &k4
                        + B6[4] * &k5)
                    + &self.sys.b * &u6);

            let xn1 = &self.state + C[0] * &k1 + C[1] * &k3 + C[2] * &k4 + C[3] * &k5;
            let xn1_ = &self.state + D[0] * &k1 + D[1] * &k3 + D[2] * &k4 + D[3] * &k5 + D[4] * &k6;

            // Take the maximum absolute error between the states of the system.
            error = (&xn1 - &xn1_).abs().max();
            let error_ratio = self.tol / error;
            // Safety factor to avoid too small step changes.
            let safety_factor = 0.95;
            if error < self.tol {
                self.h.0 = safety_factor * self.h.0 * error_ratio.powf(0.25);
                self.state = xn1;
                break;
            }
            self.h.0 = safety_factor * self.h.0 * error_ratio.powf(0.2);
        }

        // Update time before calculate the output.
        self.time.0 += self.h.0;

        let u = DVector::from_vec((self.input)(self.time));
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        Some(Rkf45 {
            time: self.time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
            error,
        })
    }
}

/// Implementation of the Iterator trait for the `Rkf45Iterator` struct
impl<'a, F> Iterator for Rkf45Iterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    type Item = Rkf45;

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
const A: [f64; 4] = [1. / 4., 3. / 8., 12. / 13., 1. / 2.];
const B21: f64 = 1. / 4.;
const B3: [f64; 2] = [3. / 32., 9. / 32.];
const B4: [f64; 3] = [1932. / 2197., -7200. / 2197., 7296. / 2197.];
const B5: [f64; 4] = [439. / 216., -8., 3680. / 513., -845. / 4104.];
const B6: [f64; 5] = [-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.];
const C: [f64; 4] = [25. / 216., 1408. / 2564., 2197. / 4101., -1. / 5.];
const D: [f64; 5] = [
    16. / 135.,
    6656. / 12_825.,
    28_561. / 56_430.,
    -9. / 50.,
    2. / 55.,
];
//////

/// Struct to hold the data of the linear system time evolution
#[derive(Debug)]
pub struct Rkf45 {
    /// Current step size
    time: Seconds,
    /// Current state
    state: Vec<f64>,
    /// Current output
    output: Vec<f64>,
    /// Current maximum absolute error
    error: f64,
}

impl Rkf45 {
    /// Get the time of the current step
    pub fn time(&self) -> Seconds {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<f64> {
        &self.output
    }

    /// Get the current maximum absolute error
    pub fn error(&self) -> f64 {
        self.error
    }
}

/// Struct for the time evolution of the linear system using the implicit
/// Radau method of order 3 with 2 steps
#[derive(Debug)]
pub struct RadauIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    /// Linear system
    sys: &'a Ss,
    /// Input function
    input: F,
    /// State vector
    state: DVector<f64>,
    /// Output vector
    output: DVector<f64>,
    /// Interval
    h: Seconds,
    /// Number of steps
    n: usize,
    /// Index
    index: usize,
    /// Tolerance
    tol: f64,
    /// Store the LU decomposition of the Jacobian matrix
    lu_jacobian: LU<f64, Dynamic, Dynamic>,
}

impl<'a, F> RadauIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
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
    pub(crate) fn new(sys: &'a Ss, u: F, x0: &[f64], h: Seconds, n: usize, tol: f64) -> Self {
        let start = DVector::from_vec(u(Seconds(0.)));
        let state = DVector::from_column_slice(x0);
        let output = &sys.c * &state + &sys.d * &start;
        // Jacobian matrix can be precomputed since it is constant for the
        // given system.
        let g = &sys.a * h.0;
        let rows = &sys.a.nrows(); // A is a square matrix.
        let identity = DMatrix::<f64>::identity(*rows, *rows);
        let j11 = &g * RADAU_A[0] - &identity;
        let j12 = &g * RADAU_A[1];
        let j21 = &g * RADAU_A[2];
        let j22 = &g * RADAU_A[3] - &identity;
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
    fn initial_step(&mut self) -> Option<Radau> {
        self.index += 1;
        Some(Radau {
            time: Seconds(0.),
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Radau order 3 with 2 step implicit method.
    #[allow(clippy::cast_precision_loss)]
    fn main_iteration(&mut self) -> Option<Radau> {
        let time = (self.index - 1) as f64 * self.h.0;
        let rows = self.sys.a.nrows();
        // k = [k1; k2] (column vector)
        let mut k = DVector::<f64>::zeros(2 * rows);
        // k sub-vectors (or block vectors) are have size (rows x 1).
        let sub_vec_size = (rows, 1);
        // Use as first guess for k1 and k2 the current state.
        k.slice_mut((0, 0), sub_vec_size).copy_from(&self.state);
        k.slice_mut((rows, 0), sub_vec_size).copy_from(&self.state);

        let u1 = DVector::from_vec((self.input)(Seconds(time + RADAU_C[0] * self.h.0)));
        let bu1 = &self.sys.b * &u1;
        let u2 = DVector::from_vec((self.input)(Seconds(time + RADAU_C[1] * self.h.0)));
        let bu2 = &self.sys.b * &u2;
        let mut f = DVector::<f64>::zeros(2 * rows);
        // Max 10 iterations.
        for _ in 0..10 {
            let k1 = k.slice((0, 0), sub_vec_size);
            let k2 = k.slice((rows, 0), sub_vec_size);

            let f1 = &self.sys.a * (&self.state + self.h.0 * (RADAU_A[0] * k1 + RADAU_A[1] * k2))
                + &bu1
                - k1;
            let f2 = &self.sys.a * (&self.state + self.h.0 * (RADAU_A[2] * k1 + RADAU_A[3] * k2))
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

            let eq = &knew.relative_eq(&k, self.tol, 0.001);
            k = knew; // Use the latest solution calculated.
            if *eq {
                break;
            }
        }
        self.state += self.h.0
            * (RADAU_B[0] * k.slice((0, 0), (rows, 1))
                + RADAU_B[1] * k.slice((rows, 0), (rows, 1)));

        let end_time = Seconds(self.index as f64 * self.h.0);
        let u = DVector::from_vec((self.input)(end_time));
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        self.index += 1;
        Some(Radau {
            time: end_time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

// Constants for Radau method.
const RADAU_A: [f64; 4] = [5. / 12., -1. / 12., 3. / 4., 1. / 4.];
const RADAU_B: [f64; 2] = [3. / 4., 1. / 4.];
const RADAU_C: [f64; 2] = [1. / 3., 1.];
//////

/// Implementation of the Iterator trait for the `RadauIterator` struct.
impl<'a, F> Iterator for RadauIterator<'a, F>
where
    F: Fn(Seconds) -> Vec<f64>,
{
    type Item = Radau;

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

/// Struct to hold the data of the linear system time evolution.
#[derive(Debug)]
pub struct Radau {
    /// Time of the current step
    time: Seconds,
    /// Current state
    state: Vec<f64>,
    /// Current output
    output: Vec<f64>,
}

impl Radau {
    /// Get the time of the current step
    pub fn time(&self) -> Seconds {
        self.time
    }

    /// Get the current state of the system
    pub fn state(&self) -> &Vec<f64> {
        &self.state
    }

    /// Get the current output of the system
    pub fn output(&self) -> &Vec<f64> {
        &self.output
    }
}
