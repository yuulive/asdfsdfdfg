//! Module for the definition of the library Error type.

use std::{error, fmt};

/// Struct to represent the error values in this library.
/// Used in `Result<T, E>` `Err(E)` variant.
pub struct Error {
    /// Internal representation of the error.
    repr: Repr,
}

/// Internal representation variants of the Error type.
#[derive(Debug)]
enum Repr {
    /// Errors that are created by this library.
    Internal(ErrorKind),
    // Add if necessary additional variants that wrap errors given by used libraries.
}

/// Enumeration of Error kinds of this library.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    /// The given system is not single input single output.
    NoSisoSystem,
    /// The given transfer function has a zero denominator.
    ZeroPolynomialDenominator,
    /// The given transfer function has no poles.
    NoPolesDenominator,
}

impl Error {
    /// Create a new internal error.
    ///
    /// # Arguments
    ///
    /// `kind` - kind of internal error
    pub(crate) fn new_internal(kind: ErrorKind) -> Self {
        Error {
            repr: Repr::Internal(kind),
        }
    }

    /// Return the kind of describing the `Error`
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self.repr {
            Repr::Internal(kind) => kind,
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
            Repr::Internal(kind) => write!(f, "Error: {:?}", kind),
        }
    }
}

impl ErrorKind {
    /// Generate the string representation of the `ErrorKind` variants.
    fn as_str(self) -> &'static str {
        match self {
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

    #[test]
    fn error_kind() {
        let err = Error::new_internal(ErrorKind::NoSisoSystem);
        assert_eq!(ErrorKind::NoSisoSystem, err.kind());

        let err = Error::new_internal(ErrorKind::ZeroPolynomialDenominator);
        assert_eq!(ErrorKind::ZeroPolynomialDenominator, err.kind());

        let err = Error::new_internal(ErrorKind::NoPolesDenominator);
        assert_eq!(ErrorKind::NoPolesDenominator, err.kind());
    }
}
