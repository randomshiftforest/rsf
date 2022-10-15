use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::{
    prelude::*,
    tests::utils::{sample_wor, time, BenchRes},
};

use super::N_REPETITIONS;

pub fn bench_offline(x: &Array2<f32>, y_true: &Array1<bool>, cb: &ConfigBuilder) -> BenchRes {
    let n1 = y_true.iter().filter(|&&a| a).count();
    let cfg = cb.build();
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let (anomalies, dt) = time(|| {
                let mut f = RSF::from_config(&cfg);
                for tree in f.iter_trees_mut() {
                    let sample = sample_wor(x, cfg.n_points);
                    tree.batch_insert(&sample);
                }
                let scores = f.batch_score(x);
                k_smallest(&scores, n1)
            });
            arr1(&[cfg.n_points as f64, pr(y_true, &anomalies), dt])
        })
        .collect();
    BenchRes::new(res)
}

pub fn bench_one_way_coordinator(
    x: &Array2<f32>,
    y_true: &Array1<bool>,
    cb: &ConfigBuilder,
) -> BenchRes {
    let n1 = y_true.iter().filter(|&&a| a).count();
    let res = (0..N_REPETITIONS)
        .into_iter()
        .map(|_| {
            let seeded_cfg = cb.clone().seed(thread_rng().gen()).build();
            let ((f, anomalies), dt) = time(|| x.outer_iter().one_way_coordinator(&seeded_cfg, n1));
            arr1(&[f.n_points() as f64, pr(y_true, &anomalies), dt])
        })
        .collect();
    BenchRes::new(res)
}

pub fn bench_two_way_par_streams(
    x: &Array2<f32>,
    y_true: &Array1<bool>,
    cb: &ConfigBuilder,
) -> BenchRes {
    let n1 = y_true.iter().filter(|&&a| a).count();
    let res = (0..N_REPETITIONS)
        .into_iter()
        .map(|_| {
            let seeded_cfg = cb.clone().seed(thread_rng().gen()).build();
            let ((f, anomalies), dt) = time(|| x.outer_iter().two_way_par_streams(&seeded_cfg, n1));
            arr1(&[f.n_points() as f64, pr(y_true, &anomalies), dt])
        })
        .collect();
    BenchRes::new(res)
}
