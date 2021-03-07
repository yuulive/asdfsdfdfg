use automatica::{poly, Ss, Tf};

/// TC2.2
#[test]
fn from_tf_to_ss() {
    let num = poly!(4.);
    let den = poly!(4., 1., 2.);
    let g = Tf::new(num, den).normalize();

    let sys = Ss::new_observability_realization(&g).unwrap();

    let expected = Ss::new_from_slice(2, 1, 1, &[0., -2., 1., -0.5], &[2., 0.], &[0., 1.], &[0.]);
    assert_eq!(expected, sys);

    assert_eq!(2, sys.dim().states());
    assert_eq!(1, sys.dim().inputs());
    assert_eq!(1, sys.dim().outputs());

    let new_tf = Tf::<f32>::new_from_siso(&sys).unwrap().normalize();

    assert_eq!(g, new_tf);
}
