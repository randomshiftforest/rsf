use crate::{prelude::*, tests::utils::save_jpeg};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use plotly::{
    common::{AxisSide, DashType, Line, Title},
    layout::{Axis, Legend},
    Layout, Plot, Scatter,
};
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use std::f32::consts::PI;

const N: usize = 1000;
const ROOT: &str = "out/examples/sine_wave";

lazy_static! {
    static ref INPUT: Array2<f32> = {
        let rng = thread_rng();
        let distr = Normal::new(0., 5.).unwrap();
        rng.sample_iter(distr)
            .take(N)
            .enumerate()
            .map(|(i, noise)| {
                let v = if (700..725).contains(&i) {
                    80.0
                } else {
                    f32::sin(i as f32 * 2.0 * PI / 100.0) * 50.0 + 100.0
                };
                v + noise
            })
            .collect::<Array1<_>>()
            .into_shape((N, 1))
            .unwrap()
    };
    static ref BASE_CB: ConfigBuilder = {
        let bb = INPUT.outer_iter().bb().unwrap();
        ConfigBuilder::default()
            .bounding_box(bb)
            .n_points(256)
            .n_trees(40)
            .window(512)
            .shingle(4)
    };
}

fn set_layout(plot: &mut Plot) {
    plot.set_layout(
        Layout::new()
            .legend(
                Legend::new()
                    .orientation(plotly::common::Orientation::Horizontal)
                    .y_anchor(plotly::common::Anchor::Bottom)
                    .x_anchor(plotly::common::Anchor::Right)
                    .y(1.02)
                    .x(1.),
            )
            .x_axis(Axis::new().title(Title::new("time")))
            .y_axis(
                Axis::new()
                    .title(Title::new("value"))
                    .range(vec![0f64, 200.]),
            )
            .y_axis2(
                Axis::new()
                    .title(Title::new("anomaly score"))
                    .overlaying("y")
                    .side(AxisSide::Right),
            ),
    );
}

#[test]
fn split_sampling() {
    let x = INPUT.clone();
    let cfg = BASE_CB.build();
    let scores: Array1<_> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_split(&cfg)
        .transform(&cfg)
        .collect();
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(0..N, x.into_iter()).show_legend(false));
    plot.add_trace(
        Scatter::new((N - scores.len())..N, scores)
            .y_axis("y2")
            .show_legend(false),
    );
    set_layout(&mut plot);
    save_jpeg(ROOT, "split_sampling", plot, 900, 450);
}

#[test]
fn joint_vs_split_reservoir_sampling() {
    let x = INPUT.clone();
    let cfg = BASE_CB.build();
    let scores_single: Array1<_> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_reservoir::<false>(&cfg)
        .transform(&cfg)
        .collect();
    let scores_multiple: Array1<_> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_reservoir::<true>(&cfg)
        .transform(&cfg)
        .collect();
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(0..N, x.into_iter())
            .name("input")
            .show_legend(false),
    );
    plot.add_trace(
        Scatter::new((N - scores_single.len())..N, scores_single)
            .line(Line::new().dash(DashType::LongDash))
            .name("joint")
            .y_axis("y2"),
    );
    plot.add_trace(
        Scatter::new((N - scores_multiple.len())..N, scores_multiple)
            .line(Line::new().dash(DashType::LongDash))
            .name("split")
            .y_axis("y2"),
    );
    set_layout(&mut plot);
    save_jpeg(ROOT, "joint_vs_split_reservoir_sampling", plot, 900, 450);
}

#[test]
fn joint_vs_split_window_sampling() {
    let x = INPUT.clone();
    let cfg = BASE_CB.build();
    let scores_single: Array1<_> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_window::<false>(&cfg)
        .transform(&cfg)
        .collect();
    let scores_multiple: Array1<_> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_window::<true>(&cfg)
        .transform(&cfg)
        .collect();
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(0..N, x.into_iter())
            .name("input")
            .show_legend(false),
    );
    plot.add_trace(
        Scatter::new((N - scores_single.len())..N, scores_single)
            .line(Line::new().dash(DashType::LongDash))
            .name("joint")
            .y_axis("y2"),
    );
    plot.add_trace(
        Scatter::new((N - scores_multiple.len())..N, scores_multiple)
            .line(Line::new().dash(DashType::LongDash))
            .name("split")
            .y_axis("y2"),
    );
    set_layout(&mut plot);
    save_jpeg(ROOT, "joint_vs_split_window_sampling", plot, 900, 450);
}

#[test]
fn varying_shingle_size() {
    let x = INPUT.clone();
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(0..N, x.clone())
            .name("input")
            .show_legend(false),
    );
    for s in [1, 4, 16] {
        let cfg = BASE_CB.clone().shingle(s).build();
        let scores: Array1<_> = x
            .outer_iter()
            .shingle(s)
            .rsf_window::<true>(&cfg)
            .transform(&cfg)
            .collect();
        plot.add_trace(
            Scatter::new((N - scores.len())..N, scores)
                .line(Line::new().dash(DashType::LongDash))
                .name(&format!("shingle_{s}"))
                .y_axis("y2"),
        );
    }
    set_layout(&mut plot);
    save_jpeg(ROOT, "varying_shingle_size", plot, 900, 450);
}
