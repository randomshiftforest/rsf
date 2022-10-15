use super::{BASE_CB, ROOT};
use crate::{
    prelude::*,
    tests::utils::{read_npz, run_globs, sample_wor, save_jpeg},
};
use plotly::{common::Title, layout::Axis as PlotAxis, Layout, Plot};
use std::{error::Error, path::PathBuf};

fn run(name: &str, path: PathBuf) -> Result<(), Box<dyn Error>> {
    let (x, _y_true) = read_npz(path);

    if x.dim().1 == 2 {
        let bb = x.outer_iter().bb().unwrap();
        let cfg = BASE_CB.clone().bounding_box(bb).build();
        let mut f = RSF::from_config(&cfg);
        let sample = sample_wor(&x, cfg.n_points);
        f.batch_insert(&sample);

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
        for tree in f.iter_trees_mut() {
            tree.add_splits(&mut layout);
        }
        let mut plot = Plot::new();
        plot.set_layout(layout);

        save_jpeg(&format!("{ROOT}/forest_splits"), name, plot, 900, 900);
    }

    Ok(())
}

#[test]
fn forest_splits_toy() {
    let _out = run_globs(run, &["in/toy/*.npz"]);
}
