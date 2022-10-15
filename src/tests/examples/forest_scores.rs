use super::{BASE_CB, ROOT};
use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, save_jpeg_stable},
};
use ndarray::prelude::*;
use plotly_stable::{
    common::{ColorScale, ColorScalePalette, Marker, Mode, Title},
    layout::Axis as PlotAxis,
    Layout, Plot, Scatter,
};
use std::{error::Error, path::PathBuf};

fn plot_tree(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (x, _y_true) = read_npz(path);

    if x.dim().1 == 2 {
        let bb = x.outer_iter().bb().unwrap();
        let cfg = BASE_CB.clone().bounding_box(bb).build();
        let scores: Array1<_> = x
            .outer_iter()
            .rsf_split(&cfg)
            .transform(&cfg)
            .map(f64::from)
            .collect();

        let mut plot = Plot::new();
        plot.set_layout(
            Layout::new()
                .x_axis(PlotAxis::new().title(Title::new("x")))
                .y_axis(PlotAxis::new().title(Title::new("y"))),
        );

        let (xs, ys): (Vec<_>, Vec<_>) = x
            .outer_iter()
            .skip(cfg.n_points)
            .map(|row| (row[0], row[1]))
            .unzip();
        plot.add_trace(
            Scatter::new(xs, ys).mode(Mode::Markers).marker(
                Marker::new()
                    .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
                    .color_array(scores.to_vec()),
            ),
        );

        save_jpeg_stable(&format!("{ROOT}/forest_scores"), name, plot, 900, 900);
    }

    Ok(())
}

#[test]
fn forest_scores_toy() {
    let _out = run_globs(plot_tree, &["in/toy/*.npz"]);
}
