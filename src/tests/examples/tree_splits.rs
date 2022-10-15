use super::ROOT;
use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, sample_wor, save_jpeg},
};
use ndarray::prelude::*;
use ndarray_rand::{RandomExt, SamplingStrategy};
use plotly::{common::Title, layout::Axis as PlotAxis, Layout, Plot, Scatter};
use rand::prelude::*;
use std::{error::Error, path::PathBuf};

fn run(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (x, y_true) = read_npz(path);

    if x.dim().1 == 2 {
        let seed: u64 = thread_rng().gen();
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
        let x = x.sample_axis_using(
            Axis(0),
            1024,
            SamplingStrategy::WithoutReplacement,
            &mut rng,
        );
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
        let y_true = y_true.sample_axis_using(
            Axis(0),
            1024,
            SamplingStrategy::WithoutReplacement,
            &mut rng,
        );

        let bb = x.outer_iter().bb().unwrap();
        let cfg = ConfigBuilder::default()
            .bounding_box(bb)
            .n_trees(2)
            .n_points(256)
            .granularity(2)
            .build();
        let mut f = RSF::from_config(&cfg);
        let sample = sample_wor(&x, cfg.n_points);
        f[1].batch_insert(&sample);

        let n1 = y_true.iter().filter(|&&a| a).count();
        let scores = f[1].batch_score(&x);
        let anomalies = k_smallest(&scores, n1);

        let bounds = BoundingBox::with_double_range(&cfg.bb).bounds;
        let mut layout = Layout::new()
            .x_axis(
                PlotAxis::new()
                    .title(Title::new("x"))
                    .range(bounds.row(0).to_vec()),
            )
            .y_axis(
                PlotAxis::new()
                    .title(Title::new("y"))
                    .range(bounds.row(1).to_vec()),
            );
        f[1].add_splits(&mut layout);
        let mut plot = Plot::new();
        plot.set_layout(layout);
        let shifted = &x + f[1].shift();
        plot.add_trace(
            Scatter::new(shifted.column(0).to_vec(), shifted.column(1).to_vec())
                .mode(plotly::common::Mode::Markers)
                .show_legend(false),
        );

        let (anomaly_xs, anomaly_ys): (Vec<_>, Vec<_>) = anomalies
            .into_iter()
            .map(|i| (shifted[(i, 0)], shifted[(i, 1)]))
            .unzip();
        plot.add_trace(
            Scatter::new(anomaly_xs, anomaly_ys)
                .mode(plotly::common::Mode::Markers)
                .show_legend(false),
        );

        save_jpeg(&format!("{ROOT}/tree_splits"), name, plot, 900, 900);
    }

    Ok(())
}

#[test]
fn tree_splits_toy() {
    let _out = run_globs(run, &["in/toy/*.npz"]);
}
