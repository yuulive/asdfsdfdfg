use num_complex::Complex;
use num_traits::{Float, FloatConst, NumCast, One, Zero};

/// Integer logarithm of a power of two.
///
/// # Arguments
///
/// * `n` - power of two
fn log2(n: usize) -> u32 {
    // core::mem::size_of::<usize>() * 8 - 1 - n.leading_zeros() as usize
    n.trailing_zeros()
}

/// Reorder the elements of the vector using a bit inversion permutation.
///
/// # Arguments
///
/// * `a` - vector with a power of two length
#[allow(non_snake_case)]
fn bit_reverse_vec<T>(a: Vec<T>) -> Vec<T> {
    let mut a = a;
    // The number of elements is a power of two.
    // The number of elements is even, iterate over half of it.
    let length = a.len();
    let half_length = length / 2;
    let bits = log2(length);

    for k in 0..half_length {
        let r = rev(k, bits);
        a.swap(k, r);
    }
    a
}

/// Reverse the last `l` bits of `k`.
///
/// # Arguments
///
/// * `k` - number on which the permutation acts.
/// * `l` - number of lower bits to reverse.
fn rev(k: usize, l: u32) -> usize {
    k.reverse_bits().rotate_left(l)
}

/// Direct Fast Fourier Transform.
///
/// # Arguments
///
/// * `a` - vector
pub(super) fn fft<T>(a: Vec<Complex<T>>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    iterative_fft(a, Transform::Direct)
}

/// Inverse Fast Fourier Transform.
///
/// # Arguments
///
/// * `y` - vector
pub(super) fn ifft<T>(y: Vec<Complex<T>>) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    iterative_fft(y, Transform::Inverse)
}

/// Extend the vector to a length that is the next power of two.
///
/// # Arguments
///
/// * `a` - vector
fn extend_to_power_of_two<T: Clone + Zero>(mut a: Vec<T>) -> Vec<T> {
    let n = a.len();
    if n.is_power_of_two() {
        a
    } else {
        let pot = n.next_power_of_two();
        a.resize(pot, T::zero());
        a
    }
}

/// Type of Fourier transform.
#[derive(Clone, Copy)]
enum Transform {
    /// Direct fast Fourier transform.
    Direct,
    /// Inverse fast Fourier transform.
    Inverse,
}

/// Iterative fast Fourier transform algorithm.
/// T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, Introduction to Algorithms, 3rd edition, 2009
///
/// # Arguments
///
/// * `a` - input vector for the transform
/// * `dir` - transform "direction" (direct or inverse)
#[allow(clippy::many_single_char_names, non_snake_case)]
fn iterative_fft<T>(a: Vec<Complex<T>>, dir: Transform) -> Vec<Complex<T>>
where
    T: Float + FloatConst + NumCast,
{
    let a = extend_to_power_of_two(a);
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut A = bit_reverse_vec(a);

    let sign = match dir {
        Transform::Direct => T::one(),
        Transform::Inverse => -T::one(),
    };

    let tau = T::TAU();
    for s in 1..=log2(n) {
        let m = 1 << s;
        let m_f = T::from(m).unwrap();
        let exp = sign * tau / m_f;
        let w_n = Complex::from_polar(T::one(), exp);
        for k in (0..n).step_by(m) {
            let mut w = Complex::one();
            for j in 0..m / 2 {
                let t = A[k + j + m / 2] * w;
                let u = A[k + j];
                A[k + j] = u + t;
                A[k + j + m / 2] = u - t;
                w = w * w_n;
            }
        }
    }

    match dir {
        Transform::Direct => A,
        Transform::Inverse => {
            let n_f = T::from(n).unwrap();
            A.iter().map(|x| x / n_f).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_bit() {
        assert_eq!(0, rev(0, 3));
        assert_eq!(4, rev(1, 3));
        assert_eq!(2, rev(2, 3));
        assert_eq!(6, rev(3, 3));
        assert_eq!(1, rev(4, 3));
        assert_eq!(5, rev(5, 3));
        assert_eq!(3, rev(6, 3));
        assert_eq!(7, rev(7, 3));
    }

    #[test]
    fn reverse_copy() {
        let a = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let b = bit_reverse_vec(a);
        let expected = vec![0, 4, 2, 6, 1, 5, 3, 7];
        assert_eq!(expected, b);
    }

    #[test]
    fn fft_iterative() {
        let one = Complex::one();
        let a = vec![one * 1., one * 0., one * 1.];
        // `a` is extended to four elements
        let f = iterative_fft(a, Transform::Direct);
        let expected = vec![one * 2., one * 0., one * 2., one * 0.];
        assert_eq!(expected, f);
    }

    #[test]
    fn fft_ifft() {
        let one = Complex::one();
        let a = vec![one * 1., one * 0., one * 1., one * 0.];
        let f = fft(a.clone());
        let a2 = ifft(f);
        assert_eq!(a, a2);
    }
}
