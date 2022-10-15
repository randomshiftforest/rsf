use super::utils::{const_copied, const_copy, save_jpeg_stable};
use crate::{
    prelude::*,
    tests::utils::{read_npz, sample_wor, save_jpeg},
};
use extended_isolation_forest::{Forest, ForestOptions};
use itertools::Itertools;
use ndarray::prelude::*;
use plotly::{
    common::{Marker, Mode, Title},
    layout::{Axis as PlotAxis, GridPattern, LayoutGrid},
    HeatMap, Layout, Plot, Scatter,
};
use std::error::Error;

const SAMPLE_SIZE: usize = 256;
const N_TREES: usize = 32;

fn run() -> Result<(), Box<dyn Error>> {
    let (mut x, y_true) = read_npz("in/examples/two_blobs.npz");
    let d = x.dim().1;
    let n1 = y_true.iter().filter(|&&a| a).count();

    if d == 2 {
        let bb = x.outer_iter().bb().unwrap();
        x.outer_iter_mut().normalise(&bb);
        let sample = sample_wor(&x, SAMPLE_SIZE);

        let xs = Array::linspace(0., 1., 1000);
        let ys = Array::linspace(0., 1., 1000);
        let xys: Vec<_> = xs
            .into_iter()
            .cartesian_product(ys.iter().cloned())
            .collect();
        let (xs, ys): (Vec<_>, Vec<_>) = xys.clone().into_iter().unzip();
        let (xs0, ys0): (Vec<_>, Vec<_>) = x.outer_iter().map(|p| (p[0], p[1])).unzip();

        // ISOLATION FOREST

        let options = ForestOptions {
            n_trees: N_TREES,
            sample_size: SAMPLE_SIZE,
            max_tree_depth: None,
            extension_level: 0,
        };
        let f = Forest::from_slice(const_copy::<2>(sample.view()).as_slice(), &options).unwrap();
        let zs1: Vec<_> = xys.iter().map(|&(x, y)| f.score(&[x, y])).collect();
        let scores1: Array1<_> = const_copied(x.outer_iter()).map(|p| f.score(&p)).collect();
        let anomalies1 = k_largest(&scores1, n1);
        let n_true1 = anomalies1.iter().filter(|&&i| y_true[i]).count();

        // - heatmap
        let mut plot1 = Plot::new();
        plot1.set_layout(
            Layout::new()
                .title(Title::new(&format!(
                    "Isolation Forest (iForest) - {n_true1}/{n1} correct"
                )))
                .grid(
                    LayoutGrid::new()
                        .rows(1)
                        .columns(2)
                        .pattern(GridPattern::Independent),
                )
                .x_axis(PlotAxis::new().title(Title::new("x")))
                .y_axis(PlotAxis::new().title(Title::new("y")))
                .x_axis2(PlotAxis::new().title(Title::new("x")))
                .y_axis2(PlotAxis::new().title(Title::new("y"))),
        );
        plot1.add_trace(
            HeatMap::new(xs.clone(), ys.clone(), zs1)
                .x_axis("x")
                .y_axis("y"),
        );
        plot1.add_trace(
            Scatter::new(xs0.clone(), ys0.clone())
                .mode(Mode::Markers)
                .marker(Marker::new().color("black"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        let (xs1, ys1): (Vec<_>, Vec<_>) = anomalies1
            .into_iter()
            .map(|i| (x[(i, 0)], x[(i, 1)]))
            .unzip();
        plot1.add_trace(
            Scatter::new(xs1, ys1)
                .mode(Mode::Markers)
                .marker(Marker::new().color("red"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        save_jpeg("out/paper/iforest", "heatmap", plot1, 900, 450);

        // tree splits
        for i in 0..4 {
            let mut layout = plotly_stable::Layout::new()
                .x_axis(
                    plotly_stable::layout::Axis::new()
                        .title(plotly_stable::common::Title::new("x"))
                        .range(vec![0., 1.])
                        .show_grid(false),
                )
                .y_axis(
                    plotly_stable::layout::Axis::new()
                        .title(plotly_stable::common::Title::new("y"))
                        .range(vec![0., 1.])
                        .show_grid(false),
                );
            f[i].add_splits(&mut layout);
            let mut plot = plotly_stable::Plot::new();
            plot.set_layout(layout);
            plot.add_trace(
                plotly_stable::Scatter::new(sample.column(0).to_vec(), sample.column(1).to_vec())
                    .mode(plotly_stable::common::Mode::Markers)
                    .show_legend(false),
            );
            save_jpeg_stable(
                "out/paper/iforest",
                &format!("tree_splits_{i}"),
                plot,
                900,
                900,
            );
        }

        // forest splits
        let mut layout = plotly_stable::Layout::new()
            .x_axis(
                plotly_stable::layout::Axis::new()
                    .title(plotly_stable::common::Title::new("x"))
                    .range(vec![0., 1.])
                    .show_grid(false),
            )
            .y_axis(
                plotly_stable::layout::Axis::new()
                    .title(plotly_stable::common::Title::new("y"))
                    .range(vec![0., 1.])
                    .show_grid(false),
            );
        for i in 0..N_TREES {
            f[i].add_splits(&mut layout);
        }
        let mut plot = plotly_stable::Plot::new();
        plot.set_layout(layout);
        plot.add_trace(plotly_stable::Scatter::new::<[f32; 0], [f32; 0]>([], []));
        save_jpeg_stable("out/paper/iforest", "forest_splits", plot, 900, 900);

        // === RANDOM SHIFT FOREST

        let cfg = ConfigBuilder::default()
            .n_points(SAMPLE_SIZE)
            .n_trees(N_TREES)
            .bounding_box(BoundingBox::unit(d))
            .build();
        let mut f = RSF::from_config(&cfg);
        f.batch_insert(&sample);
        let zs2: Vec<_> = xys
            .iter()
            .map(|&(x, y)| f.score(&arr1(&[x, y])))
            .transform(&cfg)
            .collect();
        let scores2: Array1<_> = x
            .outer_iter()
            .map(|p| f.score(&p))
            .transform(&cfg)
            .collect();
        let anomalies2 = k_largest(&scores2, n1);
        let n_true2 = anomalies2.iter().filter(|&&i| y_true[i]).count();

        // - heatmap
        let mut plot2 = Plot::new();
        plot2.set_layout(
            Layout::new()
                .title(Title::new(&format!(
                    "Random Shift Forest (RSF) - {n_true2}/{n1} correct"
                )))
                .grid(
                    LayoutGrid::new()
                        .rows(1)
                        .columns(2)
                        .pattern(GridPattern::Independent),
                )
                .x_axis(PlotAxis::new().title(Title::new("x")))
                .y_axis(PlotAxis::new().title(Title::new("y")))
                .x_axis2(PlotAxis::new().title(Title::new("x")))
                .y_axis2(PlotAxis::new().title(Title::new("y"))),
        );
        plot2.add_trace(
            HeatMap::new(xs.clone(), ys.clone(), zs2)
                .x_axis("x")
                .y_axis("y"),
        );
        plot2.add_trace(
            Scatter::new(xs0.clone(), ys0.clone())
                .mode(Mode::Markers)
                .marker(Marker::new().color("black"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        let (xs2, ys2): (Vec<_>, Vec<_>) = anomalies2
            .into_iter()
            .map(|i| (x[(i, 0)], x[(i, 1)]))
            .unzip();
        plot2.add_trace(
            Scatter::new(xs2, ys2)
                .mode(Mode::Markers)
                .marker(Marker::new().color("red"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        save_jpeg("out/paper/rsf", "heatmap", plot2, 900, 450);

        // - tree splits
        for (i, tree) in f.iter_trees_mut().enumerate().take(4) {
            let bounds = BoundingBox::with_double_range(&cfg.bb).bounds;
            let mut layout = plotly::Layout::new()
                .x_axis(
                    plotly::layout::Axis::new()
                        .title(plotly::common::Title::new("x"))
                        .range(bounds.row(0).to_vec())
                        .show_grid(false),
                )
                .y_axis(
                    plotly::layout::Axis::new()
                        .title(plotly::common::Title::new("y"))
                        .range(bounds.row(1).to_vec())
                        .show_grid(false),
                );
            tree.add_splits(&mut layout);
            let mut plot = plotly::Plot::new();
            plot.set_layout(layout);
            let shifted = &sample + tree.shift();
            plot.add_trace(
                plotly::Scatter::new(shifted.column(0).to_vec(), shifted.column(1).to_vec())
                    .mode(plotly::common::Mode::Markers)
                    .show_legend(false),
            );

            save_jpeg("out/paper/rsf", &format!("tree_splits_{i}"), plot, 900, 900);
        }

        // - forest splits
        let bounds = BoundingBox::with_double_range(&cfg.bb).bounds;
        let mut layout = plotly::Layout::new()
            .x_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::new("x"))
                    .range(bounds.row(0).to_vec())
                    .show_grid(false),
            )
            .y_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::new("y"))
                    .range(bounds.row(1).to_vec())
                    .show_grid(false),
            );
        for tree in f.iter_trees_mut() {
            tree.add_splits(&mut layout);
        }
        let mut plot = plotly::Plot::new();
        plot.set_layout(layout);
        save_jpeg("out/paper/rsf", "forest_splits", plot, 900, 900);

        // === EXTENDED ISOLATION FOREST ===

        let options = ForestOptions {
            n_trees: N_TREES,
            sample_size: SAMPLE_SIZE,
            max_tree_depth: None,
            extension_level: 1,
        };
        let f = Forest::from_slice(const_copy::<2>(sample.view()).as_slice(), &options).unwrap();
        let zs3: Vec<_> = xys.iter().map(|&(x, y)| f.score(&[x, y])).collect();
        let scores3: Array1<_> = const_copied(x.outer_iter()).map(|p| f.score(&p)).collect();
        let anomalies3 = k_largest(&scores3, n1);
        let n_true3 = anomalies3.iter().filter(|&&i| y_true[i]).count();

        // - heatmap
        let mut plot3 = Plot::new();
        plot3.set_layout(
            Layout::new()
                .title(Title::new(&format!(
                    "Extended Isolation Forest (EIF) - {n_true3}/{n1} correct"
                )))
                .grid(
                    LayoutGrid::new()
                        .rows(1)
                        .columns(2)
                        .pattern(GridPattern::Independent),
                )
                .x_axis(PlotAxis::new().title(Title::new("x")))
                .y_axis(PlotAxis::new().title(Title::new("y")))
                .x_axis2(PlotAxis::new().title(Title::new("x")))
                .y_axis2(PlotAxis::new().title(Title::new("y"))),
        );
        plot3.add_trace(HeatMap::new(xs, ys, zs3).x_axis("x").y_axis("y"));
        plot3.add_trace(
            Scatter::new(xs0, ys0)
                .mode(Mode::Markers)
                .marker(Marker::new().color("black"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        let (xs3, ys3): (Vec<_>, Vec<_>) = anomalies3
            .into_iter()
            .map(|i| (x[(i, 0)], x[(i, 1)]))
            .unzip();
        plot3.add_trace(
            Scatter::new(xs3, ys3)
                .mode(Mode::Markers)
                .marker(Marker::new().color("red"))
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2"),
        );
        save_jpeg("out/paper/eif", "heatmap", plot3, 900, 450);
    }

    Ok(())
}

#[test]
fn test() {
    run().unwrap();
}
