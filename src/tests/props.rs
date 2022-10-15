use super::{
    graphs::{input::input, DT_VALS, ET_VALS},
    utils::{paths_names, read_npz, save_jpeg, save_txt},
};
use ndarray::Array1;
use plotly::{common::Title, layout::Axis as PlotAxis, Layout, Plot, Scatter};
use std::{error::Error, fmt::Write};

const ROOT: &str = "out/props";

fn write_props(name: &str, globs: &[&str]) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    writeln!(out, "name\tpoints\tdimensions\tanomalies\tcontamination")?;
    for (path, fname) in paths_names(globs) {
        let (x, y) = read_npz(path);
        let (n, d) = x.dim();
        let n1 = y.iter().filter(|&&a| a).count();
        let c_perc = n1 as f64 / n as f64 * 100.;
        writeln!(out, "{fname}\t{n}\t{d}\t{n1}\t{c_perc:.1}%")?;
    }
    save_txt(ROOT, name, &out);
    Ok(())
}

#[test]
fn write_props_all() {
    write_props("toy", &["in/toy/*.npz"]).unwrap();
    write_props("real", &["in/real/*.npz"]).unwrap();
}

fn plot_cdf(name: &str, globs: &[&str]) {
    let mut plot = Plot::new();
    for (path, fname) in paths_names(globs) {
        let (_x, y) = read_npz(path);
        let n = y.len();
        let n1 = y.iter().filter(|&&a| a).count();
        let cdf_x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 100.).collect();
        let cdf_y: Vec<f64> = y
            .iter()
            .scan(0, |state, &anomaly| {
                if anomaly {
                    *state += 1;
                }
                Some(*state as f64 / n1 as f64 * 100.)
            })
            .collect();
        plot.add_trace(Scatter::new(cdf_x, cdf_y).name(&fname));
    }
    plot.set_layout(
        Layout::new()
            .x_axis(plotly::layout::Axis::new().title(Title::new("points (%)")))
            .y_axis(plotly::layout::Axis::new().title(Title::new("anomalies (%)"))),
    );
    save_jpeg(ROOT, name, plot, 512, 512);
}

#[test]
fn plot_cdf_all() {
    plot_cdf("toy", &["in/toy/*.npz"]);
    plot_cdf("real", &["in/real/*.npz"]);
}

fn edge_vs_anomaly(y_true: &Array1<bool>) -> Box<Scatter<f64, f64>> {
    let (xs, ys): (Vec<_>, Vec<_>) = y_true
        .iter()
        .enumerate()
        .scan(0, |n_anomalies, (i, &anomaly)| {
            if anomaly {
                *n_anomalies += 1;
            }
            Some((i as f64, *n_anomalies as f64))
        })
        .unzip();
    Scatter::new(xs, ys)
}

#[test]
fn edge_percentage_vs_anomaly_percentage() {
    let mut plot = Plot::new();
    for dt in DT_VALS {
        for et in ET_VALS {
            let (_graphs, y_true) = input("in/graph/darpa.csv", dt, et);
            let trace = edge_vs_anomaly(&y_true).name(&format!("dt{dt}_et{et}"));
            plot.add_trace(trace);
        }
    }
    plot.set_layout(
        Layout::new()
            .x_axis(PlotAxis::new().title(Title::new("#points")))
            .y_axis(PlotAxis::new().title(Title::new("#anomalies"))),
    );
    save_jpeg(ROOT, "var_dt_et", plot, 900, 450);
}
