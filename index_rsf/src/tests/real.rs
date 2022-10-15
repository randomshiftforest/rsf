use glob::glob;
use ndarray::s;
use rand::thread_rng;
use rayon::prelude::*;

use crate::{
    alg::{aabb::AABB, forest::RandShiftForest},
    tests::utils::{pr_n1, read_npz},
};

const N_TREES: usize = 32;
const N_POINTS: usize = 128;
const N_REPETITIONS: usize = 32;

#[test]
fn test_real() {
    let entries = glob("in/real/*.npz").unwrap();
    for path in entries.filter_map(Result::ok) {
        let (x, y_true) = read_npz::<f32, _>(&path);
        let aabb = AABB::from_data(&x);
        let avg_pr_n1 = (0..N_REPETITIONS)
            .into_par_iter()
            .map(|_i| {
                let mut rsf =
                    RandShiftForest::new_using(&aabb, N_TREES, N_POINTS, &mut thread_rng());
                rsf.batch_insert(&x.slice(s![0..N_POINTS, ..]));
                let scores = rsf.batch_score(&x);
                pr_n1(&y_true, &scores)
            })
            .sum::<f64>()
            / (N_REPETITIONS as f64);
        println!("{}\t{}", path.display(), avg_pr_n1);
    }
}
