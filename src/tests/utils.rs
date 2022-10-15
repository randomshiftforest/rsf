use std::{error::Error, fmt::Debug, fs, path::PathBuf, time::Instant};

use glob::glob;
use ndarray::{prelude::*, Data, ScalarOperand};
use ndarray_npy::NpzReader;
use ndarray_rand::{RandomExt, SamplingStrategy};
use num_traits::Float;
use plotly::{color::NamedColor, Plot};

pub fn read_npz<P: AsRef<std::path::Path>>(path: P) -> (Array2<f32>, Array1<bool>) {
    let f = fs::File::open(path).expect("File not found");
    let mut npz = NpzReader::new(f).expect("Invalid npz file");
    let x: Array2<f32> = npz.by_name("x.npy").expect("Could not find/parse x data");
    let y: Array1<bool> = npz.by_name("y.npy").expect("Could not find/parse y data");
    (x, y)
}

pub fn into_2d(res: Vec<Array1<f64>>) -> Result<Array2<f64>, Box<dyn Error>> {
    let d = res.first().ok_or("empty result")?.len();
    let n = res.len();
    let res_2d = res
        .into_iter()
        .flatten()
        .collect::<Array1<_>>()
        .into_shape((n, d))?;
    Ok(res_2d)
}

pub struct BenchRes {
    pub means: Array1<f64>,
    pub stds: Array1<f64>,
}

impl BenchRes {
    pub fn new(res: Vec<Array1<f64>>) -> Self {
        let res_2d = into_2d(res).unwrap();
        let means = res_2d.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = res_2d.std_axis(ndarray::Axis(0), 0.);
        Self { means, stds }
    }
}

pub fn paths_names(globs: &[&str]) -> Vec<(std::path::PathBuf, String)> {
    globs
        .iter()
        .flat_map(|pattern| {
            glob(pattern)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .map(|path| {
                    let name = path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap()
                        .to_string();
                    (path, name)
                })
        })
        .collect()
}

pub fn save_txt(root: &str, name: &str, out: &str) {
    fs::DirBuilder::new().recursive(true).create(root).unwrap();
    fs::write(format!("{root}/{name}.txt"), out).unwrap();
}

pub fn save_jpeg(root: &str, name: &str, plot: Plot, w: usize, h: usize) {
    fs::DirBuilder::new().recursive(true).create(root).unwrap();
    plot.write_image(
        format!("{root}/{name}"),
        plotly::ImageFormat::JPEG,
        w,
        h,
        2.0,
    );
}

pub fn save_jpeg_stable(root: &str, name: &str, plot: plotly_stable::Plot, w: usize, h: usize) {
    fs::DirBuilder::new().recursive(true).create(root).unwrap();
    plot.save(
        format!("{root}/{name}"),
        plotly_stable::ImageFormat::JPEG,
        w,
        h,
        2.0,
    );
}

pub fn get_color(i: usize) -> NamedColor {
    match i {
        0 => NamedColor::Orange,
        1 => NamedColor::Green,
        2 => NamedColor::Red,
        3 => NamedColor::BlueViolet,
        _ => panic!("not enough colours"),
    }
}

pub fn sample_wor<F: Float>(x: &Array2<F>, n: usize) -> Array2<F> {
    x.sample_axis(ndarray::Axis(0), n, SamplingStrategy::WithoutReplacement)
}

pub fn time<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let now = Instant::now();
    let res = f();
    let dt = now.elapsed().as_secs_f64();
    (res, dt)
}

pub fn run_globs<F, R>(f: F, globs: &[&str]) -> Vec<R>
where
    F: Fn(&str, PathBuf) -> Result<R, Box<dyn Error>>,
{
    paths_names(globs)
        .into_iter()
        .inspect(|(_path, name)| println!("{name}"))
        .map(|(path, name)| f(&name, path).unwrap())
        .collect()
}

pub fn l1_norm<F: Debug + Float + ScalarOperand, S: Data<Elem = F>>(
    arr: &ArrayBase<S, Ix1>,
) -> Array1<F> {
    let sum = arr.sum();
    if sum == F::zero() {
        Array1::zeros(arr.len())
    } else {
        arr / sum
    }
}

pub fn const_copy<const D: usize>(x: ArrayView2<f32>) -> Vec<[f32; D]> {
    x.outer_iter()
        .map(|p| {
            let mut res = [0.; D];
            for i in 0..D {
                res[i] = p[i];
            }
            res
        })
        .collect()
}

pub fn const_copied<'a, const D: usize>(
    x: impl Iterator<Item = ArrayView1<'a, f32>> + 'a,
) -> impl Iterator<Item = [f32; D]> + 'a {
    x.map(|p| {
        let mut res = [0.; D];
        for i in 0..D {
            res[i] = p[i];
        }
        res
    })
}
