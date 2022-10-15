use super::{
    bench::{bench_offline, bench_one_way_coordinator, bench_two_way_par_streams},
    SAMPLE_SIZES, SKETCH_SIZES,
};
use crate::{
    algorithm::bounding_box::BoundingBoxIter,
    prelude::ConfigBuilder,
    tests::{
        distributed::BASE_CB,
        utils::{get_color, read_npz, run_globs, save_jpeg, BenchRes},
    },
};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use plotly::{
    color::NamedColor,
    common::{Line, Title},
    layout::Legend,
    Layout, Plot, Scatter,
};
use std::{error::Error, path::PathBuf};

lazy_static! {
    static ref BASE_LAYOUT: Layout = {
        Layout::new()
            .legend(
                Legend::new()
                    .orientation(plotly::common::Orientation::Horizontal)
                    .y_anchor(plotly::common::Anchor::Bottom)
                    .x_anchor(plotly::common::Anchor::Right)
                    .y(1.)
                    .x(1.),
            )
            .x_axis(
                plotly::layout::Axis::new()
                    .title(Title::new("sample size"))
                    .range(vec![0f64, 2048.]),
            )
    };
}

fn run<B>(bencher: B, _name: &str, path: PathBuf) -> Result<(Plot, Plot), Box<dyn Error>>
where
    B: Fn(&Array2<f32>, &Array1<bool>, &ConfigBuilder) -> BenchRes,
{
    let mut size_plot = Plot::new();
    let mut pr_plot = Plot::new();

    let (x, y_true) = read_npz(&path);
    let bb = x.outer_iter().bb().ok_or("no bounding box")?;
    let base_cb = BASE_CB.clone().bounding_box(bb).n_machines(16);

    // distributed
    for (s, &sketch_size) in SKETCH_SIZES.iter().enumerate() {
        let (y_size, y_pr): (Vec<_>, Vec<_>) = SAMPLE_SIZES
            .iter()
            .map(|&n_points| {
                let cb = base_cb.clone().sketch_size(sketch_size).n_points(n_points);
                let bench_res = bencher(&x, &y_true, &cb);
                (bench_res.means[0], bench_res.means[1])
            })
            .unzip();
        size_plot.add_trace(
            Scatter::new(SAMPLE_SIZES, y_size)
                .name(&format!("sketch-{sketch_size}"))
                .line(Line::new().color(get_color(s))),
        );
        pr_plot.add_trace(
            Scatter::new(SAMPLE_SIZES, y_pr)
                .name(&format!("sketch-{sketch_size}"))
                .line(Line::new().color(get_color(s))),
        );
    }

    // offline
    let (y_size, y_pr): (Vec<_>, Vec<_>) = SAMPLE_SIZES
        .iter()
        .map(|&n_points| {
            let cb = base_cb.clone().n_points(n_points);
            let offline_res = bench_offline(&x, &y_true, &cb);
            (offline_res.means[0], offline_res.means[1])
        })
        .unzip();
    size_plot.add_trace(
        Scatter::new(SAMPLE_SIZES, y_size)
            .name("offline")
            .line(Line::new().color(NamedColor::DeepSkyBlue)),
    );
    pr_plot.add_trace(
        Scatter::new(SAMPLE_SIZES, y_pr)
            .name("offline")
            .line(Line::new().color(NamedColor::DeepSkyBlue)),
    );

    // layout
    size_plot.set_layout(
        BASE_LAYOUT.clone().y_axis(
            plotly::layout::Axis::new()
                .title(Title::new("avg. tree size"))
                .range(vec![0, 2048]),
        ),
    );
    pr_plot.set_layout(
        BASE_LAYOUT.clone().y_axis(
            plotly::layout::Axis::new()
                .title(Title::new("precision"))
                .range(vec![0, 1]),
        ),
    );

    Ok((size_plot, pr_plot))
}

fn run_one_way_coordinator(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (size_plot, pr_plot) = run(bench_one_way_coordinator, name, path)?;
    save_jpeg("out/one_way_coordinator/size", name, size_plot, 450, 450);
    save_jpeg("out/one_way_coordinator/pr", name, pr_plot, 450, 450);
    Ok(())
}

fn run_two_way_par_streams(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (size_plot, pr_plot) = run(bench_two_way_par_streams, name, path)?;
    save_jpeg("out/two_way_par_streams/size", name, size_plot, 450, 450);
    save_jpeg("out/two_way_par_streams/pr", name, pr_plot, 450, 450);
    Ok(())
}

#[test]
fn cmp_sample_sizes() {
    let globs = ["in/toy/*.npz", "in/real/*.npz"];
    let _out = run_globs(run_one_way_coordinator, &globs);
    let _out = run_globs(run_two_way_par_streams, &globs);
}
