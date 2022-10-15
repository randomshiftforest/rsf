use std::ops::Neg;

use ndarray::Array2;
use ndarray_linalg::{Determinant, QRInto};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::Rng;

// https://www.jmlr.org/papers/v17/blaser16a.html
pub fn random_rotation_using<R: Rng>(d: usize, rng: &mut R) -> Array2<f64> {
    let qr = Array2::<f64>::random_using((d, d), StandardNormal, rng)
        .qr_into()
        .unwrap();
    let diag = Array2::from_diag(&qr.1.diag().map(|&v| v.signum()));
    let mut m = qr.0.dot(&diag);
    if m.det().unwrap() < 0. {
        let col0neg = m.column(0).to_owned().neg();
        m.column_mut(0).assign(&col0neg);
    }
    m
}

#[cfg(test)]
mod tests {
    use ndarray_linalg::Inverse;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn rotation() {
        let r = random_rotation_using(3, &mut thread_rng());
        let r_t = r.t();
        let r_inv = r.inv().unwrap();
        let close = r_t
            .into_iter()
            .zip(r_inv)
            .all(|(t_v, inv_v)| f64::abs(t_v - inv_v) < 1e8);
        assert!(close);
    }
}
