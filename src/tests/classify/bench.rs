use extended_isolation_forest::{Forest, ForestOptions};
use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rcflib::rcf::create_rcf;

use crate::{prelude::*, tests::utils::BenchRes};

use super::{
    utils::{const_copied, const_copy},
    N_REPETITIONS,
};

pub fn eif_split_const<const E: bool, const D: usize>(
    x: &Array2<f32>,
    cfg: &Config,
) -> Array1<f32> {
    assert_eq!(D, x.dim().1, "wrong dataset dimensionality");
    let train = x.slice(s!(0..cfg.n_points, ..));
    let test = x.slice(s!(cfg.n_points.., ..));
    let options = ForestOptions {
        n_trees: cfg.n_trees,
        sample_size: cfg.n_points,
        max_tree_depth: None,
        extension_level: if E { D - 1 } else { 0 },
    };
    let f = Forest::from_slice(const_copy::<D>(train.view()).as_slice(), &options).unwrap();
    const_copied(test.outer_iter())
        .map(|p| f.score(&p) as f32)
        .collect()
}

pub fn bench_eif_split<const E: bool>(
    x: &Array2<f32>,
    y_true: &Array1<bool>,
    cfg: &Config,
    name: &str,
) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let y_pred = match name {
                "moons" | "circles" | "blobs" => eif_split_const::<E, 2>(x, cfg),
                "s-curve" | "swiss-roll" => eif_split_const::<E, 3>(x, cfg),
                "mulcross" => eif_split_const::<E, 4>(x, cfg),
                "shuttle" => eif_split_const::<E, 9>(x, cfg),
                "sat1" | "sat3" => eif_split_const::<E, 36>(x, cfg),
                "http" | "smtp" => eif_split_const::<E, 38>(x, cfg),
                "covtype" => eif_split_const::<E, 54>(x, cfg),
                _ => panic!("dataset dimensionality unkown"),
            };
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

pub fn bench_rrcf_split(x: &Array2<f32>, y_true: &Array1<bool>, cfg: &Config) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let mut f = create_rcf(
                x.dim().1,
                1,
                cfg.n_points,
                cfg.n_trees,
                thread_rng().gen(),
                false,
                false,
                false,
                false,
                0.0,
                0.0,
                1.0,
            );
            let mut points = x.outer_iter().map(|p| p.to_owned());
            points.by_ref().take(cfg.n_points).for_each(|p| {
                f.update(p.to_owned().as_slice().unwrap(), 0);
            });
            let y_pred: Array1<f32> = points
                .map(|p| f.score(p.as_slice().unwrap()) as f32)
                .collect();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

pub fn bench_rsf_split(x: &Array2<f32>, y_true: &Array1<bool>, cfg: &Config) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let y_pred = x
                .outer_iter()
                .rsf_split(cfg)
                .transform(cfg)
                .collect::<Array1<_>>();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

pub fn bench_rsf_res(x: &Array2<f32>, y_true: &Array1<bool>, cfg: &Config) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let y_pred = x
                .outer_iter()
                .rsf_reservoir::<true>(cfg)
                .transform(cfg)
                .collect::<Array1<_>>();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

pub fn bench_rsf_win(x: &Array2<f32>, y_true: &Array1<bool>, cfg: &Config) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let y_pred = x
                .outer_iter()
                .rsf_window::<true>(cfg)
                .transform(cfg)
                .collect::<Array1<_>>();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

// pub fn _bench_rsf_rot(x: &Array2<f32>, y_true: &Array1<bool>, cfg: &Config) -> BenchRes {
//     let aabb = index_rsf::alg::aabb::AABB::from_data(x);
//     let train = x.slice(s!(0..cfg.n_points, ..));
//     let test = x.slice(s!(cfg.n_points.., ..));
//     let res = (0..N_REPETITIONS)
//         .into_par_iter()
//         .map(|_| {
//             let mut index_rsf = index_rsf::alg::forest::RandShiftForest::new_using(
//                 &aabb,
//                 cfg.n_trees,
//                 cfg.n_points,
//                 &mut thread_rng(),
//             );
//             index_rsf.batch_insert(&train);
//             let y_pred: Array1<f32> = index_rsf
//                 .batch_score(&test)
//                 .into_iter()
//                 .transform(cfg)
//                 .collect();
//             arr1(&[
//                 rocauc(y_true, &y_pred),
//                 prauc(y_true, &y_pred),
//                 pr_n1(y_true, &y_pred),
//             ])
//         })
//         .collect::<Vec<_>>();
//     BenchRes::new(res)
// }
