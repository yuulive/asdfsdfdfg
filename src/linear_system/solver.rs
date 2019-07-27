use crate::linear_system::Ss;

use nalgebra::DVector;

/// Struct for the time evolution of a linear system
#[derive(Debug)]
pub struct Rk2Iterator<'a> {
    /// Linear system
    sys: &'a Ss,
    /// Input vector,
    input: DVector<f64>,
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
    /// Response to step function, using Runge-Kutta second order method
    ///
    /// # Arguments
    ///
    /// * `u` - input vector (colum mayor)
    /// * `x0` - initial state (colum mayor)
    /// * `h` - integration time interval
    /// * `n` - integration steps
    pub(crate) fn new(sys: &'a Ss, u: &[f64], x0: &[f64], h: f64, n: usize) -> Self {
        let input = DVector::from_column_slice(u);
        let state = DVector::from_column_slice(x0);
        // Calculate the output at time 0.
        let output = &sys.c * &state + &sys.d * &input;
        Self {
            sys,
            input,
            state,
            output,
            h,
            n,
            index: 0,
        }
    }
}

/// Implementation of the Iterator trait for the Rk2Iterator struct
impl<'a> Iterator for Rk2Iterator<'a> {
    type Item = Rk2;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.n {
            None
        } else if self.index == 0 {
            self.index += 1;
            // State and input at time 0.
            Some(Rk2 {
                time: 0.,
                state: self.state.as_slice().to_vec(),
                output: self.output.as_slice().to_vec(),
            })
        } else {
            // y_n+1 = y_n + 1/2(k1 + k2) + O(h^3)
            // k1 = h*f(x_n, y_n)
            // k2 = h*f(x_n + h, y_n + k1)
            //
            // x_n (time) does not explicitly appear for a linear system with
            // input a step function
            let bu = &self.sys.b * &self.input;
            let k1 = self.h * (&self.sys.a * &self.state + &bu);
            let k2 = self.h * (&self.sys.a * (&self.state + &k1) + &bu);
            self.state += 0.5 * (k1 + k2);
            self.output = &self.sys.c * &self.state + &self.sys.d * &self.input;

            self.index += 1;
            Some(Rk2 {
                time: self.index as f64 * self.h,
                state: self.state.as_slice().to_vec(),
                output: self.output.as_slice().to_vec(),
            })
        }
    }
}

/// Struct to hold the data of the linear system time evolution
#[derive(Debug)]
pub struct Rk2 {
    /// Time of the current step
    time: f64,
    /// Current state.
    state: Vec<f64>,
    /// Current output.
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
