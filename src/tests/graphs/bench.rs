use ndarray::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rcflib::rcf::create_rcf;

use crate::{
    prelude::*,
    tests::utils::{l1_norm, BenchRes},
};

use super::N_REPETITIONS;

#[derive(Clone, Copy)]
pub enum Mode {
    Normal,
    EdgeCounts,
    L1Norm,
}

fn sketch_graphs(
    graphs: &[Graph<String, String>],
    sl_cfg: &SpotLightConfig,
    mode: Mode,
) -> Vec<Array1<f32>> {
    match mode {
        Mode::Normal => graphs.iter().cloned().spotlight(sl_cfg).collect(),
        Mode::EdgeCounts => graphs
            .iter()
            .map(|g| arr1(&[g.edge_iter().count() as f32]))
            .collect(),
        Mode::L1Norm => graphs
            .iter()
            .cloned()
            .spotlight(sl_cfg)
            .map(|arr| l1_norm(&arr))
            .collect(),
    }
}

pub fn bench_rrcf_split(
    graphs: &[Graph<String, String>],
    y_true: &Array1<bool>,
    rsf_cb: &ConfigBuilder,
    sl_cfg: &SpotLightConfig,
    mode: Mode,
) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let sketches = sketch_graphs(graphs, sl_cfg, mode);
            let bb = sketches.iter().map(|sketch| sketch.view()).bb().unwrap();
            let rsf_cfg = rsf_cb.clone().bounding_box(bb).build();
            let (trn, tst) = sketches.split_at(rsf_cfg.n_points);
            let mut f = create_rcf(
                rsf_cfg.bb.d(),
                1,
                rsf_cfg.n_points,
                rsf_cfg.n_trees,
                thread_rng().gen(),
                false,
                false,
                false,
                false,
                0.0,
                0.0,
                1.0,
            );
            trn.iter().for_each(|p| f.update(p.as_slice().unwrap(), 0));
            let y_pred: Array1<f32> = tst
                .iter()
                .map(|p| f.score(p.as_slice().unwrap()) as f32)
                .collect();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect();
    BenchRes::new(res)
}

pub fn bench_rsf_split(
    graphs: &[Graph<String, String>],
    y_true: &Array1<bool>,
    rsf_cb: &ConfigBuilder,
    sl_cfg: &SpotLightConfig,
    mode: Mode,
) -> BenchRes {
    let res = (0..N_REPETITIONS)
        .into_par_iter()
        .map(|_| {
            let sketches = sketch_graphs(graphs, sl_cfg, mode);
            let bb = sketches.iter().map(|sketch| sketch.view()).bb().unwrap();
            let rsf_cfg = rsf_cb.clone().bounding_box(bb).build();
            let y_pred: Array1<f32> = sketches
                .into_iter()
                .rsf_split(&rsf_cfg)
                .transform(&rsf_cfg)
                .collect();
            arr1(&[
                rocauc(y_true, &y_pred),
                prauc(y_true, &y_pred),
                pr_n1(y_true, &y_pred),
            ])
        })
        .collect();
    BenchRes::new(res)
}
