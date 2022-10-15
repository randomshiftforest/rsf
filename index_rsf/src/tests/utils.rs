use std::{fs::File, path::Path};

use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1};
use ndarray_npy::{NpzReader, ReadableElement};
use num::Float;

pub fn read_npz<R: ReadableElement, P: AsRef<Path>>(path: P) -> (Array2<R>, Array1<bool>) {
    let f = File::open(path).expect("File not found");
    let mut npz = NpzReader::new(f).expect("Invalid npz file");
    let x: Array2<R> = npz.by_name("x.npy").expect("Could not find/parse x data");
    let y: Array1<bool> = npz.by_name("y.npy").expect("Could not find/parse y data");
    (x, y)
}

fn k_smallest<F: Float, S: Data<Elem = F>>(arr: &ArrayBase<S, Ix1>, k: usize) -> Vec<usize> {
    let mut max_args = Vec::from_iter(0..arr.len());
    max_args.sort_unstable_by(|&i1, &i2| arr[i1].partial_cmp(&arr[i2]).unwrap());
    max_args.into_iter().take(k).collect()
}

fn pr<S: Data<Elem = bool>>(y_true: &ArrayBase<S, Ix1>, anomalies: &[usize]) -> f64 {
    let n = anomalies.len();
    let tp = anomalies.iter().filter(|&&i| y_true[i]).count();
    (tp as f64) / (n as f64)
}

pub fn pr_n1<F: Float, S1: Data<Elem = bool>, S2: Data<Elem = F>>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64 {
    let m = y_pred.len() as isize;
    let y_true = y_true.slice(s![-m..]);
    let n1 = y_true.iter().filter(|&&anomaly| anomaly).count();
    let anomalies = k_smallest(y_pred, n1);
    pr(&y_true, &anomalies)
}
