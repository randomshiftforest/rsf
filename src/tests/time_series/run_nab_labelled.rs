use std::{fmt::Write, sync::mpsc::channel};

use chrono::NaiveDateTime;
use ndarray::prelude::*;
use plotly::{
    color::NamedColor,
    common::{AxisSide, Title},
    layout::{Axis, Shape, ShapeLayer, ShapeLine, ShapeType},
    Layout, Plot, Scatter,
};
use rayon::{iter::repeatn, prelude::*};

use crate::{
    prelude::*,
    tests::utils::{save_jpeg, save_txt},
};

use super::{
    input::input,
    utils::{autoperiod, WINDOWS},
    ROOT,
};

pub fn run(
    ts: &[String],
    x: Array2<f32>,
    wins: &[[String; 2]],
    cfg: &Config,
    mut layout: Layout,
) -> Plot {
    let scores: Array1<f32> = x
        .outer_iter()
        .shingle(cfg.shingle)
        .rsf_window::<true>(cfg)
        .transform(cfg)
        .collect();
    let n = x.dim().0;
    let shift = n - scores.len();
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(ts.to_vec(), Vec::from_iter(x))
            .show_legend(false)
            .name("input"),
    );
    plot.add_trace(
        Scatter::new(ts[shift..n].to_vec(), scores)
            .y_axis("y2")
            .show_legend(false)
            .name("output"),
    );
    for win in wins {
        layout.add_shape(
            Shape::new()
                .x_ref("x")
                .y_ref("paper")
                .shape_type(ShapeType::Rect)
                .x0(&win[0])
                .y0(0f64)
                .x1(&win[1])
                .y1(1f64)
                .fill_color(NamedColor::LightSalmon)
                .opacity(0.5)
                .layer(ShapeLayer::Below)
                .line(ShapeLine::new().width(0.)),
        );
    }
    plot.set_layout(
        layout
            .x_axis(Axis::new().title(Title::new("date-time")))
            .y_axis(Axis::new().title(Title::new("value")))
            .y_axis2(
                Axis::new()
                    .overlaying("y")
                    .side(AxisSide::Right)
                    .title(Title::new("anomaly score")),
            ),
    );
    plot
}

fn parse_date(date: &str) -> i64 {
    NaiveDateTime::parse_from_str(date, "%Y-%m-%d %H:%M:%S%.6f")
        .unwrap()
        .timestamp()
}

#[test]
fn run_nab_labelled() {
    let mut out = String::new();
    writeln!(out, "class\tname\t#points\t#anomalies\tperiod").unwrap();
    let (tx, rx) = channel::<String>();

    let entries: Vec<_> = WINDOWS.iter().collect();
    let m = entries.len();
    entries
        .into_par_iter()
        .zip(repeatn(tx, m))
        .for_each(|(((class, name), wins), tx)| {
            let n_trees = 64;
            let n_points = 512;
            let window = 2048;

            let (ts, x) = input(class, name);
            let values: Vec<_> = x.iter().cloned().take(1024).collect();
            let period = autoperiod(values).unwrap();
            if period == 1 {
                println!("{class} {name} - no period");
                return;
            }
            // let shingle = usize::max(128, period);
            let shingle = period;

            if wins.is_empty() {
                println!("{class} {name} - no labels");
                return;
            }
            let win0_start = parse_date(&wins[0][0]);
            let win0_idx = ts
                .iter()
                .map(|t| parse_date(t))
                .position(|t| t > win0_start)
                .unwrap();
            if win0_idx < window {
                println!("{class} {name} - window too early");
                return;
            }

            let bb = x.outer_iter().bb().unwrap();
            let cfg = ConfigBuilder::default()
                .bounding_box(bb)
                .n_trees(n_trees)
                .n_points(n_points)
                .window(window)
                .shingle(shingle)
                .build();
            let plot = run(&ts, x, wins.as_slice(), &cfg, Layout::new());
            save_jpeg(&format!("{ROOT}/{class}"), name, plot, 900, 450);
            tx.send(format!(
                "{class}\t{name}\t{}\t{}\t{}\n",
                ts.len(),
                wins.len(),
                period
            ))
            .unwrap();
        });

    while let Ok(text) = rx.recv() {
        write!(out, "{text}").unwrap();
    }
    save_txt(ROOT, "stats", &out);
}
