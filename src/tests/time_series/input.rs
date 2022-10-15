use ndarray::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct NabRec {
    timestamp: String,
    value: f32,
}

pub fn input(class: &str, name: &str) -> (Vec<String>, Array2<f32>) {
    let path = format!("in/nab/data/{class}/{name}.csv");
    let rdr = csv::Reader::from_path(path).expect("data file not found");
    let (timestamps, values): (Vec<String>, Vec<f32>) = rdr
        .into_deserialize::<NabRec>()
        .map(|rec| rec.unwrap())
        .map(|rec| (rec.timestamp, rec.value))
        .unzip();
    let n = values.len();
    let values = Array1::from_vec(values).into_shape((n, 1)).unwrap();
    (timestamps, values)
}
