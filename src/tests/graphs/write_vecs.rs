use ndarray::prelude::*;

use crate::{adapter::spotlight_alt::SpotLightAltIter, prelude::*};

use super::input::input;

#[test]
fn run() {
    let dt = 3600;
    let et = 60;
    println!("aggregation: {dt}s");
    println!("edge threshold: {et}");

    let (graphs, y_true) = input("in/graph/darpa.csv", dt, et);
    let n = graphs.len();
    let n1 = y_true.iter().filter(|&&anomaly| anomaly).count();
    println!("graphs: {n}, anomalies: {n1}");

    let sl_cfg = SpotLightConfig::new(50, 0.2, 0.2);
    let sketches: Vec<Array1<f32>> = graphs.iter().cloned().spotlight_alt(&sl_cfg).collect();
    println!("label\t\tzero\tnonzero");
    for (sketch, anomalous) in sketches.iter().zip(y_true) {
        let n_zero = sketch.iter().filter(|&&v| v == 0.).count();
        let n_nonzero = sketch.len() - n_zero;
        let label = if anomalous { "anomaly" } else { "normal" };
        println!("{label}\t\t{n_zero}\t{n_nonzero}");
    }
}
