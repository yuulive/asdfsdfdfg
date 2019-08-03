use crate::linear_system::Ss;

use nalgebra::{DMatrix, DVector};

/// Struct for the time evolution of a linear system
#[derive(Debug)]
pub struct Rk2Iterator<'a> {
    /// Linear system
    sys: &'a Ss,
    /// Input function
    input: fn(f64) -> Vec<f64>,
    /// State vector.
    state: DVector<f64>,
    /// Output vector.
    output: DVector<f64>,
    /// Interval.
    h: f64,
    /// Number of steps.
    n: usize,
    /// Index.
    index: usize,
}

impl<'a> Rk2Iterator<'a> {
    /// Create the solver for a Runge-Kutta second order method
    ///
    /// # Arguments
    ///
    /// * `u` - input function that returns a vector (colum vector)
    /// * `x0` - initial state (colum vector)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub(crate) fn new(sys: &'a Ss, u: fn(f64) -> Vec<f64>, x0: &[f64], h: f64, n: usize) -> Self {
        let start = DVector::from_vec(u(0.0));
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
        }
    }

    /// Intial step (time 0) of the rk2 solver.
    /// It contains the initial state and the calculated inital output
    /// at the constructor
    fn initial_step(&mut self) -> Option<Rk2> {
        self.index += 1;
        // State and output at time 0.
        Some(Rk2 {
            time: 0.,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    /// Runge-Kutta order 2 method
    fn main_iteration(&mut self) -> Option<Rk2> {
        // y_n+1 = y_n + 1/2(k1 + k2) + O(h^3)
        // k1 = h*f(t_n, y_n)
        // k2 = h*f(t_n + h, y_n + k1)
        let time = self.index as f64 * self.h;
        let u = DVector::from_vec((self.input)(time));
        let uh = DVector::from_vec((self.input)(time + self.h));
        let bu = &self.sys.b * &u;
        let buh = &self.sys.b * &uh;
        let k1 = self.h * (&self.sys.a * &self.state + &bu);
        let k2 = self.h * (&self.sys.a * (&self.state + &k1) + &buh);
        self.state += 0.5 * (k1 + k2);
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        self.index += 1;
        Some(Rk2 {
            time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

/// Implementation of the Iterator trait for the Rk2Iterator struct
impl<'a> Iterator for Rk2Iterator<'a> {
    type Item = Rk2;

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

/// Struct to hold the data of the linear system time evolution
#[derive(Debug)]
pub struct Rk2 {
    /// Time of the current step
    time: f64,
    /// Current state
    state: Vec<f64>,
    /// Current output
    output: Vec<f64>,
}

impl Rk2 {
    /// Get the time of the current step
    pub fn time(&self) -> f64 {
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
pub struct Rkf45Iterator<'a> {
    /// Linear system
    sys: &'a Ss,
    /// Input function
    input: fn(f64) -> Vec<f64>,
    /// State vector.
    state: DVector<f64>,
    /// Output vector.
    output: DVector<f64>,
    /// Interval.
    h: f64,
    /// Time limit of the evaluation
    limit: f64,
    /// Time
    time: f64,
    /// Tollerance
    tol: f64,
    /// Is initial step
    initial_step: bool,
}

impl<'a> Rkf45Iterator<'a> {
    /// Create a solver using Runge-Kutta-Fehlberg method
    ///
    /// # Arguments
    ///
    /// * `u` - input function (colum vector)
    /// * `x0` - initial state (colum vector)
    /// * `h` - integration time interval
    /// * `limit` - time limit of the evaluation
    /// * `tol` - error tollerance
    pub(crate) fn new(
        sys: &'a Ss,
        u: fn(f64) -> Vec<f64>,
        x0: &[f64],
        h: f64,
        limit: f64,
        tol: f64,
    ) -> Self {
        let start = DVector::from_vec(u(0.0));
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
            time: 0.,
            tol,
            initial_step: true,
        }
    }

    /// Intial step (time 0) of the rkf45 solver.
    /// It contains the initial state and the calculated inital output
    /// at the constructor
    fn initial_step(&mut self) -> Option<Rkf45> {
        self.initial_step = false;
        Some(Rkf45 {
            time: 0.,
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
            let u2 = DVector::from_vec((self.input)(self.time + self.h * A[0]));
            let u3 = DVector::from_vec((self.input)(self.time + self.h * A[1]));
            let u4 = DVector::from_vec((self.input)(self.time + self.h * A[2]));
            let u5 = DVector::from_vec((self.input)(self.time + self.h));
            let u6 = DVector::from_vec((self.input)(self.time + self.h * A[3]));

            let k1 = self.h * (&self.sys.a * &self.state + &self.sys.b * &u1);
            let k2 = self.h * (&self.sys.a * (&self.state + B21 * &k1) + &self.sys.b * &u2);
            let k3 = self.h
                * (&self.sys.a * (&self.state + B3[0] * &k1 + B3[1] * &k2) + &self.sys.b * &u3);
            let k4 = self.h
                * (&self.sys.a * (&self.state + B4[0] * &k1 + B4[1] * &k2 + B4[2] * &k3)
                    + &self.sys.b * &u4);
            let k5 = self.h
                * (&self.sys.a
                    * (&self.state + B5[0] * &k1 + B5[1] * &k2 + B5[2] * &k3 + B5[3] * &k4)
                    + &self.sys.b * &u5);
            let k6 = self.h
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
                self.h = safety_factor * self.h * error_ratio.powf(0.25);
                self.state = xn1;
                break;
            }
            self.h = safety_factor * self.h * error_ratio.powf(0.2);
        }

        // Update time before calculate the output.
        self.time += self.h;

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

/// Implementation of the Iterator trait for the Rkf45Iterator struct
impl<'a> Iterator for Rkf45Iterator<'a> {
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

// Coefficients of th Butcher table of rkf45 method.
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

/// Struct to hold the data of the linar system time evolution
#[derive(Debug)]
pub struct Rkf45 {
    /// Current step size
    time: f64,
    /// Current state
    state: Vec<f64>,
    /// Current output
    output: Vec<f64>,
    /// Current maximum absolute error
    error: f64,
}

impl Rkf45 {
    /// Get the time of the current step
    pub fn time(&self) -> f64 {
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

#[derive(Debug)]
pub struct RadauIterator<'a> {
    sys: &'a Ss,
    input: fn(f64) -> Vec<f64>,
    state: DVector<f64>,
    output: DVector<f64>,
    h: f64,
    n: usize,
    index: usize,
    tol: f64,
    /// Store the inverted jacobian
    inv_jacobian: DMatrix<f64>,
}

impl<'a> RadauIterator<'a> {
    pub(crate) fn new(
        sys: &'a Ss,
        u: fn(f64) -> Vec<f64>,
        x0: &[f64],
        h: f64,
        n: usize,
        tol: f64,
    ) -> Self {
        let start = DVector::from_vec(u(0.0));
        let state = DVector::from_column_slice(x0);
        let output = &sys.c * &state + &sys.d * &start;
        // Jacobian matrix can be precomputed since it is constant for the
        // given system.
        let g = &sys.a * h;
        let rows = &sys.a.nrows(); // A is a square matrix.
        let i = DMatrix::<f64>::identity(*rows, *rows);
        let j11 = &g * RADAU_A[0] - &i;
        let j12 = &g * RADAU_A[1];
        let j21 = &g * RADAU_A[2];
        let j22 = &g * RADAU_A[3] - &i;
        let mut j = DMatrix::zeros(2 * *rows, 2 * *rows);
        // Copy the sub matrices into the the Jacobian.
        let sub_matrix_size = (*rows, *rows);
        j.slice_mut((0, 0), sub_matrix_size).copy_from(&j11);
        j.slice_mut((0, *rows), sub_matrix_size).copy_from(&j12);
        j.slice_mut((*rows, 0), sub_matrix_size).copy_from(&j21);
        j.slice_mut((*rows, *rows), sub_matrix_size).copy_from(&j22);

        Self {
            sys,
            input: u,
            state,
            output,
            h,
            n,
            index: 0,
            tol,
            inv_jacobian: j.try_inverse().unwrap(),
        }
    }

    fn initial_step(&mut self) -> Option<Radau> {
        self.index += 1;
        Some(Radau {
            time: 0.,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }

    fn main_iteration(&mut self) -> Option<Radau> {
        let time = self.index as f64 * self.h;
        let rows = self.sys.a.nrows();
        // k = [k1; k2]
        let mut k = DVector::<f64>::zeros(2 * rows);
        // Use as first guess for k1 and k2 the current state.
        k.slice_mut((0, 0), (rows, 1)).copy_from(&self.state);
        k.slice_mut((rows, 0), (rows, 1)).copy_from(&self.state);

        let u1 = DVector::from_vec((self.input)(time + RADAU_C[0] * self.h));
        let bu1 = &self.sys.b * &u1;
        let u2 = DVector::from_vec((self.input)(time + RADAU_C[1] * self.h));
        let bu2 = &self.sys.b * &u2;
        // Max 10 iterations.
        for _ in 0..10 {
            let k1 = k.slice((0, 0), (rows, 1));
            let k2 = k.slice((rows, 0), (rows, 1));

            let f1 = &self.sys.a * (&self.state + self.h * (RADAU_A[0] * &k1 + RADAU_A[1] * &k2))
                + &bu1
                - &k1;
            let f2 = &self.sys.a * (&self.state + self.h * (RADAU_A[2] * &k1 + RADAU_A[3] * &k2))
                + &bu2
                - &k2;
            let mut f = DVector::<f64>::zeros(2 * rows);
            f.slice_mut((0, 0), (rows, 1)).copy_from(&f1);
            f.slice_mut((rows, 0), (rows, 1)).copy_from(&f2);

            let dk = -&self.inv_jacobian * &f;
            let knew = &k + &dk;

            let eq = &knew.relative_eq(&k, self.tol, 0.001);
            if *eq {
                k = knew;
                break;
            }

            k = knew;
        }
        self.state += self.h
            * (RADAU_B[0] * &k.slice((0, 0), (rows, 1))
                + RADAU_B[1] * k.slice((rows, 0), (rows, 1)));

        let u = DVector::from_vec((self.input)(time + self.h));
        self.output = &self.sys.c * &self.state + &self.sys.d * &u;

        self.index += 1;
        Some(Radau {
            time,
            state: self.state.as_slice().to_vec(),
            output: self.output.as_slice().to_vec(),
        })
    }
}

const RADAU_A: [f64; 4] = [5. / 12., -1. / 12., 3. / 4., 1. / 4.];
const RADAU_B: [f64; 2] = [3. / 4., 1. / 4.];
const RADAU_C: [f64; 2] = [1. / 3., 1.];

impl<'a> Iterator for RadauIterator<'a> {
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

#[derive(Debug)]
pub struct Radau {
    time: f64,
    state: Vec<f64>,
    output: Vec<f64>,
}

impl Radau {
    /// Get the time of the current step
    pub fn time(&self) -> f64 {
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