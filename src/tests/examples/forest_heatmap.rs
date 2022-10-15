use super::{BASE_CB, ROOT};
use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, sample_wor, save_jpeg},
};
use itertools::Itertools;
use ndarray::{arr1, Array};
use plotly::{common::Title, layout::Axis as PlotAxis, HeatMap, Layout, Plot};
use std::{error::Error, path::PathBuf};

fn run(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (x, _y_true) = read_npz(path);

    if x.dim().1 == 2 {
        let bb = x.outer_iter().bb().unwrap();
        let cfg = BASE_CB.clone().bounding_box(bb).build();
        let mut f = RSF::from_config(&cfg);
        let sample = sample_wor(&x, cfg.n_points);
        f.batch_insert(&sample);

        let xs = Array::linspace(cfg.bb.bounds[(0, 0)], cfg.bb.bounds[(0, 1)], 1000);
        let ys = Array::linspace(cfg.bb.bounds[(1, 0)], cfg.bb.bounds[(1, 1)], 1000);

        let xys: Vec<_> = xs
            .into_iter()
            .cartesian_product(ys.iter().cloned())
            .collect();
        let zs: Vec<_> = xys
            .iter()
            .map(|&(x, y)| f.score(&arr1(&[x, y])))
            .transform(&cfg)
            .collect();

        let mut plot = Plot::new();
        plot.set_layout(
            Layout::new()
                .x_axis(PlotAxis::new().title(Title::new("x")))
                .y_axis(PlotAxis::new().title(Title::new("y"))),
        );
        let (xs, ys): (Vec<_>, Vec<_>) = xys.into_iter().unzip();
        plot.add_trace(HeatMap::new(xs, ys, zs));
        save_jpeg(&format!("{ROOT}/forest_heatmap"), name, plot, 900, 900)
    }

    Ok(())
}

#[test]
fn forest_heatmap_toy() {
    let _out = run_globs(run, &["in/toy/*.npz"]);
}
