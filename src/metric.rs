use classifier_measures::{pr_auc, roc_auc};
use ndarray::{s, ArrayBase, Data, Ix1};
use num_traits::Float;

pub fn rocauc<F: Float, S1: Data<Elem = bool>, S2: Data<Elem = F>>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64 {
    let n = y_true.len();
    let m = y_pred.len();
    let offset = n - m;
    roc_auc(0..m, |i| (y_true[offset + i], y_pred[i]))
        .unwrap()
        .to_f64()
        .unwrap()
}

pub fn prauc<F: Float, S1: Data<Elem = bool>, S2: Data<Elem = F>>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64 {
    let n = y_true.len();
    let m = y_pred.len();
    let offset = n - m;
    pr_auc(0..m, |i| (y_true[offset + i], y_pred[i]))
        .unwrap()
        .to_f64()
        .unwrap()
}

pub fn k_smallest<F: Float, S: Data<Elem = F>>(arr: &ArrayBase<S, Ix1>, k: usize) -> Vec<usize> {
    let mut min_args = Vec::from_iter(0..arr.len());
    min_args.sort_unstable_by(|&i1, &i2| arr[i1].partial_cmp(&arr[i2]).unwrap());
    min_args.into_iter().take(k).collect()
}

pub fn k_largest<F: Float, S: Data<Elem = F>>(arr: &ArrayBase<S, Ix1>, k: usize) -> Vec<usize> {
    let mut max_args = Vec::from_iter(0..arr.len());
    max_args.sort_unstable_by(|&i1, &i2| arr[i1].partial_cmp(&arr[i2]).unwrap().reverse());
    max_args.into_iter().take(k).collect()
}

pub fn pr_n1<F: Float, S1: Data<Elem = bool>, S2: Data<Elem = F>>(
    y_true: &ArrayBase<S1, Ix1>,
    y_pred: &ArrayBase<S2, Ix1>,
) -> f64 {
    let m = y_pred.len() as isize;
    let y_true = y_true.slice(s![-m..]);
    let n1 = y_true.iter().filter(|&&anomaly| anomaly).count();
    let anomalies = k_largest(y_pred, n1);
    pr(&y_true, &anomalies)
}

pub fn pr<S: Data<Elem = bool>>(y_true: &ArrayBase<S, Ix1>, anomalies: &[usize]) -> f64 {
    let n = anomalies.len();
    let tp = anomalies.iter().filter(|&&i| y_true[i]).count();
    (tp as f64) / (n as f64)
}

pub fn rc<S: Data<Elem = bool>>(y_true: &ArrayBase<S, Ix1>, anomalies: &[usize]) -> f64 {
    let n1 = y_true.iter().filter(|&&a| a).count();
    let tp = anomalies.iter().filter(|&&i| y_true[i]).count();
    (tp as f64) / (n1 as f64)
}

pub fn exp_bst_path_length(n: usize) -> f32 {
    let h = |i: f32| i.ln() + 0.577_215_7;
    let c = |n: f32| 2.0 * h(n - 1.0) - (2.0 * (n - 1.0) / n);
    c(n as f32)
}
