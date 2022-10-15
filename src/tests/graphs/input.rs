use std::path::Path;

use itertools::Itertools;
use ndarray::prelude::*;
use serde::Deserialize;

use crate::prelude::*;

#[derive(Deserialize)]
struct EdgeRec {
    src: String,
    dst: String,
    sec: usize,
    anamoly: bool,
}

fn read_csv<P: AsRef<Path>>(path: P) -> impl Iterator<Item = EdgeRec> {
    csv::Reader::from_path(path)
        .expect("file not found")
        .into_deserialize::<EdgeRec>()
        .map(|rec| rec.unwrap())
}

pub fn input<P: AsRef<Path>>(
    path: P,
    dt: usize,
    et: usize,
) -> (Vec<Graph<String, String>>, Array1<bool>) {
    let graphs: Vec<Graph<String, String>> = read_csv(&path)
        .group_by(|rec| rec.sec / dt)
        .into_iter()
        .map(|(_key, group)| {
            let edges: Vec<(String, String, f32)> =
                group.map(|rec| (rec.src, rec.dst, 1.0)).collect();
            Graph::new(edges)
        })
        .collect();
    let y_true: Array1<bool> = read_csv(&path)
        .group_by(|rec| rec.sec / dt)
        .into_iter()
        .map(|(_key, group)| group.filter(|rec| rec.anamoly).count() >= et)
        .collect();
    (graphs, y_true)
}
