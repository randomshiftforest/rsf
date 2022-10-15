use crate::{
    prelude::*,
    tests::utils::{get_color, read_npz, run_globs, save_jpeg, time, BenchRes},
};
use ndarray::prelude::*;
use plotly::{
    common::{Line, Title},
    layout::{Axis, Legend},
    Layout, Plot,
};
use rand::{thread_rng, Rng};
use std::{error::Error, path::PathBuf};

const ROOT: &str = "out/one_way_coordinator/scalability";
const N_REPETITIONS: usize = 32;

pub fn bench_pre_filter_unpar(
    x: &Array2<f32>,
    y_true: &Array1<bool>,
    cb: &ConfigBuilder,
) -> BenchRes {
    let n1 = y_true.iter().filter(|&&a| a).count();
    let res = (0..N_REPETITIONS)
        .into_iter()
        .map(|_| {
            let seeded_cfg = cb.clone().seed(thread_rng().gen()).build();
            // let pool = ThreadPoolBuilder::new()
            //     .num_threads(seeded_cfg.n_machines)
            //     .build()
            //     .unwrap();
            // pool.install(|| {
            //     let ((f, anomalies), dt) = time(|| x.outer_iter().pre_filter(&seeded_cfg, n1));
            //     arr1(&[f.n_points() as f64, pr(y_true, &anomalies), dt])
            // })
            let ((f, anomalies), dt) = time(|| x.outer_iter().one_way_coordinator(&seeded_cfg, n1));
            arr1(&[f.n_points() as f64, pr(y_true, &anomalies), dt])
        })
        .collect::<Vec<_>>();
    BenchRes::new(res)
}

fn run(_name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let machine_sizes = [16, 32, 64, 128];
    let sample_sizes = [256, 512, 1024, 2048];

    let mut size_plot = Plot::new();
    let mut precision_plot = Plot::new();
    let mut time_plot = Plot::new();

    let (x, y_true) = read_npz(&path);
    let bb = x.outer_iter().bb().ok_or("no bounding box")?;

    // benchmark
    for (s, sample_size) in sample_sizes.into_iter().enumerate() {
        let (size, (precision, time)): (Vec<_>, (Vec<_>, Vec<_>)) = machine_sizes
            .iter()
            .map(|&machine_size| {
                let cb = ConfigBuilder::default()
                    .bounding_box(bb.clone())
                    .n_points(sample_size)
                    .n_trees(32)
                    .granularity(4)
                    .n_machines(machine_size);
                let res = bench_pre_filter_unpar(&x, &y_true, &cb);
                (res.means[0], (res.means[1], res.means[2]))
            })
            .unzip();

        size_plot.add_trace(
            plotly::Scatter::new(machine_sizes, size)
                .name(&format!("sample_size_{sample_size}"))
                .line(Line::new().color(get_color(s))),
        );
        precision_plot.add_trace(
            plotly::Scatter::new(machine_sizes, precision)
                .name(&format!("sample_size_{sample_size}"))
                .line(Line::new().color(get_color(s))),
        );
        time_plot.add_trace(
            plotly::Scatter::new(machine_sizes, time)
                .name(&format!("sample_size_{sample_size}"))
                .line(Line::new().color(get_color(s))),
        );
    }

    // layout
    let base_layout = Layout::new()
        .x_axis(Axis::new().title(Title::new("#machines")))
        .legend(
            Legend::new()
                .orientation(plotly::common::Orientation::Horizontal)
                .y_anchor(plotly::common::Anchor::Bottom)
                .x_anchor(plotly::common::Anchor::Right)
                .y(1.)
                .x(1.),
        );
    size_plot.set_layout(
        base_layout
            .clone()
            .y_axis(Axis::new().title(Title::new("avg. tree size"))),
    );
    precision_plot.set_layout(
        base_layout.clone().y_axis(
            Axis::new()
                .title(Title::new("precision"))
                .range(vec![0., 1.]),
        ),
    );
    time_plot
        .set_layout(base_layout.y_axis(Axis::new().title(Title::new("running time (seconds)"))));

    // save
    save_jpeg(ROOT, "size", size_plot, 900, 450);
    save_jpeg(ROOT, "precision", precision_plot, 900, 450);
    save_jpeg(ROOT, "time", time_plot, 900, 450);

    Ok(())
}

#[test]
fn cmp_machine_sizes() {
    let globs = ["in/real/http.npz"];
    let _out = run_globs(run, &globs);
}
