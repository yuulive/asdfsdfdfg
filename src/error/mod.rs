//! Module for the definition of the library Error type.

use std::{error, fmt};

/// Struct to represent the error values in this library.
pub struct Error {
    /// Internal representation of the error.
    repr: Repr,
}

#[derive(Debug)]
enum Repr {
    Internal(ErrorKind),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ErrorKind {
    NoSisoSystem,
    ZeroPolynomialDenominator,
    NoPolesDenominator,
}

impl Error {
    pub(crate) fn new_internal(kind: ErrorKind) -> Self {
        Error {
            repr: Repr::Internal(kind),
        }
    }
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.repr {
            Repr::Internal(kind) => write!(f, "{}", kind.as_str()),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.repr {
            Repr::Internal(kind) => write!(
                f,
                "Error: {}, [file: {}, line: {}]",
                kind.as_str(),
                file!(),
                line!()
            ),
        }
    }
}

impl ErrorKind {
    fn as_str(&self) -> &'static str {
        match *self {
            ErrorKind::NoSisoSystem => "Linear system is not Single Input Single Output",
            ErrorKind::ZeroPolynomialDenominator => {
                "Transfer functions cannot have zero polynomial denominator"
            }
            ErrorKind::NoPolesDenominator => "Denominator has no poles",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_error() {
        let err = Error::new_internal(ErrorKind::NoSisoSystem);
        assert!(!err.to_string().is_empty());
        assert!(!format!("{:?}", err).is_empty());
        assert_eq!(ErrorKind::NoSisoSystem.as_str(), err.to_string());

        let err = Error::new_internal(ErrorKind::ZeroPolynomialDenominator);
        assert!(!err.to_string().is_empty());
        assert!(!format!("{:?}", err).is_empty());
        assert_eq!(
            ErrorKind::ZeroPolynomialDenominator.as_str(),
            err.to_string()
        );

        let err = Error::new_internal(ErrorKind::NoPolesDenominator);
        assert!(!err.to_string().is_empty());
        assert!(!format!("{:?}", err).is_empty());
        assert_eq!(ErrorKind::NoPolesDenominator.as_str(), err.to_string());
    }
}
