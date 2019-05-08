pub mod linear_system;
pub mod polynomial;
pub mod transfer_function;

/// Trait for the implementation of object evaluation
pub trait Eval<T> {
    /// Evaluate the polynomial at the value x
    ///
    /// # Arguments
    ///
    /// * `x` - Value at which the polynomial is evaluated
    fn eval(&self, x: &T) -> T;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
